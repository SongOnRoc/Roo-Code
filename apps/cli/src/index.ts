/**
 * @roo-code/cli - Command Line Interface for Roo Code
 */

import { Command } from "commander"
import fs from "fs"
import os from "os"
import path from "path"
import { fileURLToPath } from "url"

import { worktreeService, worktreeIncludeService } from "@roo-code/core"
import {
	type ProviderName,
	type ReasoningEffortExtended,
	isProviderName,
	reasoningEffortsExtended,
} from "@roo-code/types"
import { setLogger } from "@roo-code/vscode-shim"

import { ExtensionHost } from "./extension-host.js"
import { getEnvVarName, getApiKeyFromEnv, getDefaultExtensionPath } from "./utils.js"

const DEFAULTS = {
	mode: "code",
	reasoningEffort: "medium" as const,
	model: "anthropic/claude-sonnet-4.5",
}

const REASONING_EFFORTS = [...reasoningEffortsExtended, "unspecified", "disabled"]

const __dirname = path.dirname(fileURLToPath(import.meta.url))

/**
 * Generate a random alphanumeric suffix for branch/folder names
 */
function generateRandomSuffix(length = 5): string {
	const chars = "abcdefghijklmnopqrstuvwxyz0123456789"
	let result = ""
	for (let i = 0; i < length; i++) {
		result += chars.charAt(Math.floor(Math.random() * chars.length))
	}
	return result
}

/**
 * Generate worktree defaults (branch name and path).
 */
function generateWorktreeDefaults(workspacePath: string): { suggestedBranch: string; suggestedPath: string } {
	const suffix = generateRandomSuffix()
	const projectName = path.basename(workspacePath)
	const dotRooPath = path.join(os.homedir(), ".roo")
	const suggestedPath = path.join(dotRooPath, "RooCode", "worktrees", `${projectName}-${suffix}`)

	return {
		suggestedBranch: `worktree/roo-${suffix}`,
		suggestedPath,
	}
}

const program = new Command()

program.name("roo").description("Roo Code CLI - Run the Roo Code agent from the command line").version("0.1.0")

/**
 * Create a worktree using @roo-code/core directly (no extension needed).
 */
async function createWorktreeForTask(
	workspacePath: string,
	options: { verbose: boolean; debug: boolean },
): Promise<{ path: string; branch: string } | null> {
	const gitInstalled = await worktreeService.checkGitInstalled()

	if (!gitInstalled) {
		console.error("[CLI] Error: git is not installed or not in PATH")
		return null
	}

	const isGitRepo = await worktreeService.checkGitRepo(workspacePath)

	if (!isGitRepo) {
		console.error("[CLI] Error: workspace is not a git repository")
		return null
	}

	const defaults = generateWorktreeDefaults(workspacePath)

	if (options.debug) {
		console.log(`[CLI] Worktree defaults - Branch: ${defaults.suggestedBranch}, Path: ${defaults.suggestedPath}`)
	}

	console.log(`[CLI] Creating worktree at ${defaults.suggestedPath}...`)

	const result = await worktreeService.createWorktree(workspacePath, {
		path: defaults.suggestedPath,
		branch: defaults.suggestedBranch,
		createNewBranch: true,
	})

	if (!result.success || !result.worktree) {
		console.error(`[CLI] Error creating worktree: ${result.message}`)
		return null
	}

	// Copy .worktreeinclude files if present.
	try {
		const copiedItems = await worktreeIncludeService.copyWorktreeIncludeFiles(workspacePath, result.worktree.path)

		if (copiedItems.length > 0) {
			console.log(`[CLI] Copied ${copiedItems.length} item(s) from .worktreeinclude`)
		}
	} catch (error) {
		// Log but don't fail.
		if (options.verbose) {
			console.log(`[CLI] Warning: Failed to copy .worktreeinclude files: ${error}`)
		}
	}

	return {
		path: result.worktree.path,
		branch: result.worktree.branch,
	}
}

program
	.command("run", { isDefault: true })
	.argument("<prompt>", "The prompt/task to execute")
	.option("-w, --workspace <path>", "Workspace path to operate in", process.cwd())
	.option("-e, --extension <path>", "Path to the extension bundle directory")
	.option("-v, --verbose", "Enable verbose output (show VSCode and extension logs)", false)
	.option("-d, --debug", "Enable debug output (includes detailed debug information)", false)
	.option("-x, --exit-on-complete", "Exit the process when the task completes (useful for testing)", false)
	.option("-y, --yes", "Auto-approve all prompts (non-interactive mode)", false)
	.option("-k, --api-key <key>", "API key for the LLM provider (defaults to ANTHROPIC_API_KEY env var)")
	.option("-p, --provider <provider>", "API provider (anthropic, openai, openrouter, etc.)", "openrouter")
	.option("-m, --model <model>", "Model to use", DEFAULTS.model)
	.option("-M, --mode <mode>", "Mode to start in (code, architect, ask, debug, etc.)", DEFAULTS.mode)
	.option(
		"-r, --reasoning-effort <effort>",
		"Reasoning effort level (unspecified, disabled, none, minimal, low, medium, high, xhigh)",
		DEFAULTS.reasoningEffort,
	)
	.option("-W, --worktree", "Run the task in a new worktree (allows parallel CLI instances)", false)
	.action(
		async (
			prompt: string,
			options: {
				workspace: string
				extension?: string
				verbose: boolean
				debug: boolean
				exitOnComplete: boolean
				yes: boolean
				apiKey?: string
				provider: ProviderName
				model?: string
				mode?: string
				reasoningEffort?: ReasoningEffortExtended | "unspecified" | "disabled"
				worktree: boolean
			},
		) => {
			// Default is quiet mode - suppress VSCode shim logs unless verbose
			// or debug is specified.
			if (!options.verbose && !options.debug) {
				setLogger({
					info: () => {},
					warn: () => {},
					error: () => {},
					debug: () => {},
				})
			}

			const extensionPath = options.extension || getDefaultExtensionPath(__dirname)
			const apiKey = options.apiKey || getApiKeyFromEnv(options.provider)
			const workspacePath = path.resolve(options.workspace)

			if (!apiKey) {
				console.error(
					`[CLI] Error: No API key provided. Use --api-key or set the appropriate environment variable.`,
				)
				console.error(`[CLI] For ${options.provider}, set ${getEnvVarName(options.provider)}`)
				process.exit(1)
			}

			if (!fs.existsSync(workspacePath)) {
				console.error(`[CLI] Error: Workspace path does not exist: ${workspacePath}`)
				process.exit(1)
			}

			if (!isProviderName(options.provider)) {
				console.error(`[CLI] Error: Invalid provider: ${options.provider}`)
				process.exit(1)
			}

			if (options.reasoningEffort && !REASONING_EFFORTS.includes(options.reasoningEffort)) {
				console.error(
					`[CLI] Error: Invalid reasoning effort: ${options.reasoningEffort}, must be one of: ${REASONING_EFFORTS.join(", ")}`,
				)
				process.exit(1)
			}

			console.log(`[CLI] Mode: ${options.mode || "default"}`)
			console.log(`[CLI] Reasoning Effort: ${options.reasoningEffort || "default"}`)
			console.log(`[CLI] Provider: ${options.provider}`)
			console.log(`[CLI] Model: ${options.model || "default"}`)
			console.log(`[CLI] Workspace: ${workspacePath}`)

			// Track worktree info to use its path for the task
			let createdWorktree: { path: string; branch: string } | null = null
			let taskWorkspacePath = workspacePath

			try {
				// If --worktree flag is set, create a worktree first using @roo-code/core directly
				// This avoids the singleton issue by not needing multiple ExtensionHost activations
				if (options.worktree) {
					console.log("[CLI] Worktree mode enabled - creating isolated worktree for task...")

					createdWorktree = await createWorktreeForTask(workspacePath, options)

					if (!createdWorktree) {
						console.error("[CLI] Failed to create worktree, aborting")
						process.exit(1)
					}

					console.log(`[CLI] Worktree created at: ${createdWorktree.path}`)
					console.log(`[CLI] Branch: ${createdWorktree.branch}`)

					// Use the worktree path for the task
					taskWorkspacePath = createdWorktree.path
				}

				// Create the host with the appropriate workspace (original or worktree)
				const taskHost = new ExtensionHost({
					mode: options.mode || DEFAULTS.mode,
					reasoningEffort: options.reasoningEffort === "unspecified" ? undefined : options.reasoningEffort,
					apiProvider: options.provider,
					apiKey,
					model: options.model || DEFAULTS.model,
					workspacePath: taskWorkspacePath,
					extensionPath: path.resolve(extensionPath),
					verbose: options.debug,
					quiet: !options.verbose && !options.debug,
					nonInteractive: options.yes,
				})

				// Run the task
				await taskHost.activate()
				await taskHost.runTask(prompt)
				await taskHost.dispose()

				if (options.exitOnComplete) {
					process.exit(0)
				}
			} catch (error) {
				console.error("[CLI] Error:", error instanceof Error ? error.message : String(error))

				if (options.debug && error instanceof Error) {
					console.error(error.stack)
				}

				process.exit(1)
			}
		},
	)

program.parse()
