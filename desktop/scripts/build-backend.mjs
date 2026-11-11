import { spawnSync } from "node:child_process";
import { existsSync, rmSync } from "node:fs";
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
  console.error("Run `uv sync --extra speech --extra build` from the repository root,");
  console.error("or set DANGLISH_PYTHON to a Python with the project installed.");
  process.exit(1);
}

const destination = path.join(root, "desktop", "resources", "backend");
rmSync(destination, { recursive: true, force: true });

console.log("Building desktop backend with PyInstaller ...");
const result = spawnSync(
  python,
  [
    "-m",
    "PyInstaller",
    path.join(root, "packaging", "danglish-backend.spec"),
    "--noconfirm",
    "--clean",
    "--distpath",
    destination,
    "--workpath",
    path.join(root, "build", "pyinstaller"),
  ],
  { cwd: root, stdio: "inherit", env: process.env }
);

if (result.error) {
  console.error(result.error.message);
  process.exit(1);
}
if (result.status !== 0) {
  console.error("PyInstaller failed. Install it with `uv sync --extra build`.");
  process.exit(result.status ?? 1);
}

const executable = path.join(
  destination,
  "danglish-backend",
  isWindows ? "danglish-backend.exe" : "danglish-backend"
);
if (!existsSync(executable)) {
  console.error(`Build finished but ${executable} is missing.`);
  process.exit(1);
}

if (process.platform === "darwin") {
  // Ad-hoc signing lets the unsigned development app launch the sidecar on macOS.
  // Real distribution builds should use a Developer ID signature and notarization.
  console.log("Ad-hoc signing the macOS backend ...");
  const sign = spawnSync(
    "codesign",
    ["--force", "--sign", "-", executable],
    { stdio: "inherit" }
  );
  if (sign.status !== 0) {
    console.error("codesign failed; the packaged sidecar may be blocked by Gatekeeper.");
  }
}

console.log(`Backend ready: ${executable}`);
