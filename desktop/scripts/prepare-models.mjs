import { spawnSync } from "node:child_process";
import { existsSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(__dirname, "..", "..");
const isWindows = process.platform === "win32";
const python =
  process.env.DANGLISH_PYTHON ||
  (isWindows
    ? path.join(root, ".venv", "Scripts", "python.exe")
    : path.join(root, ".venv", "bin", "python"));

if (!existsSync(python)) {
  console.error(`Python environment not found: ${python}`);
  console.error("Run `uv sync --extra speech --extra convert --extra build` first.");
  process.exit(1);
}

const args = [
  path.join(root, "scripts", "bundle_models.py"),
  "--output",
  path.join(root, "desktop", "resources", "models"),
];
const mtSource = path.join(root, "models", "opus-mt-da-en-int8");
if (existsSync(mtSource)) args.push("--mt-source", mtSource);
if (process.env.DANGLISH_TTS_VOICE) {
  args.push("--voice", process.env.DANGLISH_TTS_VOICE);
}

console.log("Bundling models for the desktop app ...");
const result = spawnSync(python, args, { cwd: root, stdio: "inherit", env: process.env });
if (result.error) {
  console.error(result.error.message);
  process.exit(1);
}
process.exit(result.status ?? 1);
