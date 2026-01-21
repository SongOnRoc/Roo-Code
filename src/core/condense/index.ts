import Anthropic from "@anthropic-ai/sdk"
import crypto from "crypto"

import { TelemetryService } from "@roo-code/telemetry"

import { t } from "../../i18n"
import { ApiHandler } from "../../api"
import { ApiMessage } from "../task-persistence/apiMessages"
import { maybeRemoveImageBlocks } from "../../api/transform/image-cleaning"
import { findLast } from "../../shared/array"
import { supportPrompt } from "../../shared/support-prompt"

/**
 * Checks if a message contains tool_result blocks.
 * For native tools protocol, user messages with tool_result blocks require
 * corresponding tool_use blocks from the previous assistant turn.
 */
function hasToolResultBlocks(message: ApiMessage): boolean {
	if (message.role !== "user" || typeof message.content === "string") {
		return false
	}
	return message.content.some((block) => block.type === "tool_result")
}

/**
 * Gets the tool_use blocks from a message.
 */
function getToolUseBlocks(message: ApiMessage): Anthropic.Messages.ToolUseBlock[] {
	if (message.role !== "assistant" || typeof message.content === "string") {
		return []
	}
	return message.content.filter((block) => block.type === "tool_use") as Anthropic.Messages.ToolUseBlock[]
}

/**
 * Gets the tool_result blocks from a message.
 */
function getToolResultBlocks(message: ApiMessage): Anthropic.ToolResultBlockParam[] {
	if (message.role !== "user" || typeof message.content === "string") {
		return []
	}
	return message.content.filter((block): block is Anthropic.ToolResultBlockParam => block.type === "tool_result")
}

/**
 * Finds a tool_use block by ID in a message.
 */
function findToolUseBlockById(message: ApiMessage, toolUseId: string): Anthropic.Messages.ToolUseBlock | undefined {
	if (message.role !== "assistant" || typeof message.content === "string") {
		return undefined
	}
	return message.content.find(
		(block): block is Anthropic.Messages.ToolUseBlock => block.type === "tool_use" && block.id === toolUseId,
	)
}

/**
 * Gets reasoning blocks from a message's content array.
 * Task stores reasoning as {type: "reasoning", text: "..."} blocks,
 * which convertToR1Format and convertToZAiFormat already know how to extract.
 */
function getReasoningBlocks(message: ApiMessage): Anthropic.Messages.ContentBlockParam[] {
	if (message.role !== "assistant" || typeof message.content === "string") {
		return []
	}
	// Filter for reasoning blocks and cast to ContentBlockParam (the type field is compatible)
	return message.content.filter((block) => (block as any).type === "reasoning") as any[]
}

/**
 * Result of getKeepMessagesWithToolBlocks
 */
export type KeepMessagesResult = {
	keepMessages: ApiMessage[]
	toolUseBlocksToPreserve: Anthropic.Messages.ToolUseBlock[]
	// Reasoning blocks from the preceding assistant message, needed for DeepSeek/Z.ai
	// when tool_use blocks are preserved. Task stores reasoning as {type: "reasoning", text: "..."}
	// blocks, and convertToR1Format/convertToZAiFormat already extract these.
	reasoningBlocksToPreserve: Anthropic.Messages.ContentBlockParam[]
}

/**
 * Extracts tool_use blocks that need to be preserved to match tool_result blocks in keepMessages.
 * Checks ALL kept messages for tool_result blocks and searches backwards through the condensed
 * region (bounded by N_MESSAGES_TO_KEEP) to find the matching tool_use blocks by ID.
 * These tool_use blocks will be appended to the summary message to maintain proper pairing.
 *
 * Also extracts reasoning blocks from messages containing preserved tool_uses, which are required
 * by DeepSeek and Z.ai for interleaved thinking mode. Without these, the API returns a 400 error
 * "Missing reasoning_content field in the assistant message".
 * See: https://api-docs.deepseek.com/guides/thinking_mode#tool-calls
 *
 * @param messages - The full conversation messages
 * @param keepCount - The number of messages to keep from the end
 * @returns Object containing keepMessages, tool_use blocks, and reasoning blocks to preserve
 */
export function getKeepMessagesWithToolBlocks(messages: ApiMessage[], keepCount: number): KeepMessagesResult {
	if (messages.length <= keepCount) {
		return { keepMessages: messages, toolUseBlocksToPreserve: [], reasoningBlocksToPreserve: [] }
	}

	const startIndex = messages.length - keepCount
	const keepMessages = messages.slice(startIndex)

	const toolUseBlocksToPreserve: Anthropic.Messages.ToolUseBlock[] = []
	const reasoningBlocksToPreserve: Anthropic.Messages.ContentBlockParam[] = []
	const preservedToolUseIds = new Set<string>()

	// Check ALL kept messages for tool_result blocks
	for (const keepMsg of keepMessages) {
		if (!hasToolResultBlocks(keepMsg)) {
			continue
		}

		const toolResults = getToolResultBlocks(keepMsg)

		for (const toolResult of toolResults) {
			const toolUseId = toolResult.tool_use_id

			// Skip if we've already found this tool_use
			if (preservedToolUseIds.has(toolUseId)) {
				continue
			}

			// Search backwards through the condensed region (bounded)
			const searchStart = startIndex - 1
			const searchEnd = Math.max(0, startIndex - N_MESSAGES_TO_KEEP)
			const messagesToSearch = messages.slice(searchEnd, searchStart + 1)

			// Find the message containing this tool_use
			const messageWithToolUse = findLast(messagesToSearch, (msg) => {
				return findToolUseBlockById(msg, toolUseId) !== undefined
			})

			if (messageWithToolUse) {
				const toolUse = findToolUseBlockById(messageWithToolUse, toolUseId)!
				toolUseBlocksToPreserve.push(toolUse)
				preservedToolUseIds.add(toolUseId)

				// Also preserve reasoning blocks from that message
				const reasoning = getReasoningBlocks(messageWithToolUse)
				reasoningBlocksToPreserve.push(...reasoning)
			}
		}
	}

	return {
		keepMessages,
		toolUseBlocksToPreserve,
		reasoningBlocksToPreserve,
	}
}

export const N_MESSAGES_TO_KEEP = 3
export const MIN_CONDENSE_THRESHOLD = 5 // Minimum percentage of context window to trigger condensing
export const MAX_CONDENSE_THRESHOLD = 100 // Maximum percentage of context window to trigger condensing

const SUMMARY_PROMPT = supportPrompt.default.CONDENSE

export type SummarizeResponse = {
	messages: ApiMessage[] // The messages after summarization
	summary: string // The summary text; empty string for no summary
	cost: number // The cost of the summarization operation
	newContextTokens?: number // The number of tokens in the context for the next API request
	error?: string // Populated iff the operation fails: error message shown to the user on failure (see Task.ts)
	condenseId?: string // The unique ID of the created Summary message, for linking to condense_context clineMessage
}

/**
 * Summarizes the conversation messages using an LLM call
 *
 * @param {ApiMessage[]} messages - The conversation messages
 * @param {ApiHandler} apiHandler - The API handler to use for token counting.
 * @param {string} systemPrompt - The system prompt for API requests, which should be considered in the context token count
 * @param {string} taskId - The task ID for the conversation, used for telemetry
 * @param {boolean} isAutomaticTrigger - Whether the summarization is triggered automatically
 * @returns {SummarizeResponse} - The result of the summarization operation (see above)
 */
/**
 * Summarizes the conversation messages using an LLM call
 *
 * @param {ApiMessage[]} messages - The conversation messages
 * @param {ApiHandler} apiHandler - The API handler to use for token counting (fallback if condensingApiHandler not provided)
 * @param {string} systemPrompt - The system prompt for API requests (fallback if customCondensingPrompt not provided)
 * @param {string} taskId - The task ID for the conversation, used for telemetry
 * @param {number} prevContextTokens - The number of tokens currently in the context, used to ensure we don't grow the context
 * @param {boolean} isAutomaticTrigger - Whether the summarization is triggered automatically
 * @param {string} customCondensingPrompt - Optional custom prompt to use for condensing
 * @param {ApiHandler} condensingApiHandler - Optional specific API handler to use for condensing
 * @param {boolean} useNativeTools - Whether native tools protocol is being used (requires tool_use/tool_result pairing)
 * @returns {SummarizeResponse} - The result of the summarization operation (see above)
 */
export async function summarizeConversation(
	messages: ApiMessage[],
	apiHandler: ApiHandler,
	systemPrompt: string,
	taskId: string,
	prevContextTokens: number,
	isAutomaticTrigger?: boolean,
	customCondensingPrompt?: string,
	condensingApiHandler?: ApiHandler,
	useNativeTools?: boolean,
): Promise<SummarizeResponse> {
	TelemetryService.instance.captureContextCondensed(
		taskId,
		isAutomaticTrigger ?? false,
		!!customCondensingPrompt?.trim(),
		!!condensingApiHandler,
	)

	const response: SummarizeResponse = { messages, cost: 0, summary: "" }

	// We no longer preserve "kept" messages after condense. Instead, we:
	// - Summarize everything except the last N messages
	// - Embed the last N messages verbatim as <system-reminder> blocks inside the final summary message
	// - Tag all previous messages with condenseParent so only the final message is effective for the API
	//
	// Keep the existing minimum threshold semantics: require at least (N_MESSAGES_TO_KEEP + 2) messages.
	// With N_MESSAGES_TO_KEEP=3, this means at least 5 messages.
	if (messages.length <= N_MESSAGES_TO_KEEP + 1) {
		const error = t("common:errors.condense_not_enough_messages")
		return { ...response, error }
	}

	const reminders = messages.slice(-N_MESSAGES_TO_KEEP)

	// Messages to summarize for the LLM call. This can include the reminder messages; they are
	// also embedded verbatim into the final condensed output as <system-reminder> blocks.
	const messagesToSummarize = getMessagesSinceLastSummary(messages)

	// Defensive: if we somehow have nothing meaningful to summarize.
	if (messagesToSummarize.length <= 1) {
		const error = t("common:errors.condense_not_enough_messages")
		return { ...response, error }
	}

	// Check if there's a recent summary in the messages we're embedding as reminders
	const recentSummaryExists = reminders.some((message: ApiMessage) => message.isSummary)

	if (recentSummaryExists) {
		const error = t("common:errors.condensed_recently")
		return { ...response, error }
	}

	const finalRequestMessage: Anthropic.MessageParam = {
		role: "user",
		content: "Summarize the conversation so far, as described in the prompt instructions.",
	}

	const requestMessages = maybeRemoveImageBlocks([...messagesToSummarize, finalRequestMessage], apiHandler).map(
		({ role, content }) => ({ role, content }),
	)

	// Note: this doesn't need to be a stream, consider using something like apiHandler.completePrompt
	// Use custom prompt if provided and non-empty, otherwise use the default SUMMARY_PROMPT
	const promptToUse = customCondensingPrompt?.trim() ? customCondensingPrompt.trim() : SUMMARY_PROMPT

	// Use condensing API handler if provided, otherwise use main API handler
	let handlerToUse = condensingApiHandler || apiHandler

	// Check if the chosen handler supports the required functionality
	if (!handlerToUse || typeof handlerToUse.createMessage !== "function") {
		console.warn(
			"Chosen API handler for condensing does not support message creation or is invalid, falling back to main apiHandler.",
		)

		handlerToUse = apiHandler // Fallback to the main, presumably valid, apiHandler

		// Ensure the main apiHandler itself is valid before this point or add another check.
		if (!handlerToUse || typeof handlerToUse.createMessage !== "function") {
			// This case should ideally not happen if main apiHandler is always valid.
			// Consider throwing an error or returning a specific error response.
			console.error("Main API handler is also invalid for condensing. Cannot proceed.")
			// Return an appropriate error structure for SummarizeResponse
			const error = t("common:errors.condense_handler_invalid")
			return { ...response, error }
		}
	}

	const stream = handlerToUse.createMessage(promptToUse, requestMessages)

	let summary = ""
	let cost = 0
	let outputTokens = 0

	for await (const chunk of stream) {
		if (chunk.type === "text") {
			summary += chunk.text
		} else if (chunk.type === "usage") {
			// Record final usage chunk only
			cost = chunk.totalCost ?? 0
			outputTokens = chunk.outputTokens ?? 0
		}
	}

	summary = summary.trim()

	if (summary.length === 0) {
		const error = t("common:errors.condense_failed")
		return { ...response, cost, error }
	}

	// Build the final "condensed context" message.
	// After condense, the effective API history MUST be exactly one final role:"user" message
	// whose content is 4 text blocks:
	// 1) preface paragraph + summary text
	// 2-4) <system-reminder> blocks for the last 3 messages (role+ts+raw content JSON)
	const prefaceParagraph =
		"Condensing conversation context. The summary below captures the key information from the prior conversation."

	const summaryTextBlock: Anthropic.Messages.TextBlockParam = {
		type: "text",
		text: `${prefaceParagraph}\n\n${summary}`,
	}

	const reminderBlocks: Anthropic.Messages.TextBlockParam[] = reminders.map((msg) => {
		const tsValue = msg.ts ?? null
		const rawContentJson = JSON.stringify(msg.content)
		return {
			type: "text",
			text: `<system-reminder role="${msg.role}" ts="${tsValue}">${rawContentJson}</system-reminder>`,
		}
	})

	// Ensure we always produce exactly 3 reminder blocks. If fewer exist (should be rare), pad with explicit empties.
	while (reminderBlocks.length < N_MESSAGES_TO_KEEP) {
		reminderBlocks.unshift({
			type: "text",
			text: `<system-reminder role="unknown" ts="null">null</system-reminder>`,
		})
	}

	const summaryContent: Anthropic.Messages.ContentBlockParam[] = [summaryTextBlock, ...reminderBlocks].slice(
		0,
		1 + N_MESSAGES_TO_KEEP,
	)

	// Generate a unique condenseId for this summary
	const condenseId = crypto.randomUUID()

	const lastTs = messages[messages.length - 1]?.ts ?? Date.now()
	const summaryMessage: ApiMessage = {
		role: "user",
		content: summaryContent,
		ts: lastTs + 1,
		isSummary: true,
		condenseId,
	}

	// NON-DESTRUCTIVE CONDENSE:
	// Tag all prior messages (including the last N reminders) with condenseParent so only
	// the final summaryMessage is effective. Preserve nested condense by not overwriting
	// an existing condenseParent.
	const taggedMessages = messages.map((msg) => {
		if (!msg.condenseParent) {
			return { ...msg, condenseParent: condenseId }
		}
		return msg
	})

	const newMessages = [...taggedMessages, summaryMessage]

	// Count the tokens in the context for the next API request.
	// The next API request will include the system prompt plus the final condensed user message.
	const systemPromptMessage: ApiMessage = { role: "user", content: systemPrompt }
	const contextBlocks = [systemPromptMessage, summaryMessage].flatMap((message) =>
		typeof message.content === "string" ? [{ text: message.content, type: "text" as const }] : message.content,
	)

	const newContextTokens = await apiHandler.countTokens(contextBlocks)
	if (newContextTokens >= prevContextTokens) {
		const error = t("common:errors.condense_context_grew")
		return { ...response, cost, error }
	}
	return { messages: newMessages, summary, cost, newContextTokens, condenseId }
}

/* Returns the list of all messages since the last summary message, including the summary. Returns all messages if there is no summary. */
export function getMessagesSinceLastSummary(messages: ApiMessage[]): ApiMessage[] {
	let lastSummaryIndexReverse = [...messages].reverse().findIndex((message) => message.isSummary)

	if (lastSummaryIndexReverse === -1) {
		return messages
	}

	const lastSummaryIndex = messages.length - lastSummaryIndexReverse - 1
	const messagesSinceSummary = messages.slice(lastSummaryIndex)

	return messagesSinceSummary
}

/**
 * Filters the API conversation history to get the "effective" messages to send to the API.
 * Messages with a condenseParent that points to an existing summary are filtered out,
 * as they have been replaced by that summary.
 * Messages with a truncationParent that points to an existing truncation marker are also filtered out,
 * as they have been hidden by sliding window truncation.
 *
 * This allows non-destructive condensing and truncation where messages are tagged but not deleted,
 * enabling accurate rewind operations while still sending condensed/truncated history to the API.
 *
 * @param messages - The full API conversation history including tagged messages
 * @returns The filtered history that should be sent to the API
 */
export function getEffectiveApiHistory(messages: ApiMessage[]): ApiMessage[] {
	// Collect all condenseIds of summaries that exist in the current history
	const existingSummaryIds = new Set<string>()
	// Collect all truncationIds of truncation markers that exist in the current history
	const existingTruncationIds = new Set<string>()

	for (const msg of messages) {
		if (msg.isSummary && msg.condenseId) {
			existingSummaryIds.add(msg.condenseId)
		}
		if (msg.isTruncationMarker && msg.truncationId) {
			existingTruncationIds.add(msg.truncationId)
		}
	}

	// Filter out messages whose condenseParent points to an existing summary
	// or whose truncationParent points to an existing truncation marker.
	// Messages with orphaned parents (summary/marker was deleted) are included.
	//
	// Additionally, when a summary exists, ONLY that summary should remain effective;
	// any non-summary messages with the same condenseParent must be filtered.
	return messages.filter((msg) => {
		// Filter out condensed messages if their summary exists
		if (msg.condenseParent && existingSummaryIds.has(msg.condenseParent)) {
			return false
		}
		// If a summary exists, any non-summary message that shares its condenseParent should be hidden.
		// This handles the new condense output where *all* previous messages are tagged.
		if (!msg.isSummary) {
			for (const summaryId of existingSummaryIds) {
				if (msg.condenseParent === summaryId) {
					return false
				}
			}
		}
		// Filter out truncated messages if their truncation marker exists
		if (msg.truncationParent && existingTruncationIds.has(msg.truncationParent)) {
			return false
		}
		return true
	})
}

/**
 * Cleans up orphaned condenseParent and truncationParent references after a truncation operation (rewind/delete).
 * When a summary message or truncation marker is deleted, messages that were tagged with its ID
 * should have their parent reference cleared so they become active again.
 *
 * This function should be called after any operation that truncates the API history
 * to ensure messages are properly restored when their summary or truncation marker is deleted.
 *
 * @param messages - The API conversation history after truncation
 * @returns The cleaned history with orphaned condenseParent and truncationParent fields cleared
 */
export function cleanupAfterTruncation(messages: ApiMessage[]): ApiMessage[] {
	// Collect all condenseIds of summaries that still exist
	const existingSummaryIds = new Set<string>()
	// Collect all truncationIds of truncation markers that still exist
	const existingTruncationIds = new Set<string>()

	for (const msg of messages) {
		if (msg.isSummary && msg.condenseId) {
			existingSummaryIds.add(msg.condenseId)
		}
		if (msg.isTruncationMarker && msg.truncationId) {
			existingTruncationIds.add(msg.truncationId)
		}
	}

	// Clear orphaned parent references for messages whose summary or truncation marker was deleted
	return messages.map((msg) => {
		let needsUpdate = false

		// Check for orphaned condenseParent
		if (msg.condenseParent && !existingSummaryIds.has(msg.condenseParent)) {
			needsUpdate = true
		}

		// Check for orphaned truncationParent
		if (msg.truncationParent && !existingTruncationIds.has(msg.truncationParent)) {
			needsUpdate = true
		}

		if (needsUpdate) {
			// Create a new object without orphaned parent references
			const { condenseParent, truncationParent, ...rest } = msg
			const result: ApiMessage = rest as ApiMessage

			// Keep condenseParent if its summary still exists
			if (condenseParent && existingSummaryIds.has(condenseParent)) {
				result.condenseParent = condenseParent
			}

			// Keep truncationParent if its truncation marker still exists
			if (truncationParent && existingTruncationIds.has(truncationParent)) {
				result.truncationParent = truncationParent
			}

			return result
		}
		return msg
	})
}
