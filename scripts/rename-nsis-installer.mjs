import { execFile } from "node:child_process";
import { readdir, rename, rm } from "node:fs/promises";
import { readFile } from "node:fs/promises";
import { dirname, join } from "node:path";
import { promisify } from "node:util";
import { fileURLToPath } from "node:url";

const execFileAsync = promisify(execFile);

const root = join(dirname(fileURLToPath(import.meta.url)), "..");
const config = JSON.parse(await readFile(join(root, "src-tauri", "tauri.conf.json"), "utf8"));
const directory = join(root, "src-tauri", "target", "release", "bundle", "nsis");
const prefix = `${config.productName}_${config.version}_`;
const names = (await readdir(directory)).filter((name) => name.endsWith("-setup.exe"));
const generated = names.filter((name) => name.startsWith(prefix));
if (generated.length === 0 && !names.some((name) => name.startsWith(`Model-SDK-Workbench_${config.version}_`))) {
  throw new Error(`No NSIS installer found in ${directory}`);
}
for (const name of generated) {
  const archSetup = name.slice(prefix.length);
  const targetName = `Model-SDK-Workbench_${config.version}_${archSetup}`;
  const source = join(directory, name);
  const target = join(directory, targetName);
  if (name === targetName) continue;
  if (names.includes(targetName)) await rm(target);
  await rename(source, target);
}
const outputs = (await readdir(directory)).filter(
  (name) => name.startsWith(`Model-SDK-Workbench_${config.version}_`) && name.endsWith("-setup.exe"),
);
if (outputs.length === 0) throw new Error(`No NSIS installer found in ${directory}`);
for (const name of outputs) {
  const target = join(directory, name);
  await execFileAsync("icacls", [target, "/setintegritylevel", "Medium"]);
  console.log(target);
}
