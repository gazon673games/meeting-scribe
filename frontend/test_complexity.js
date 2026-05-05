/**
 * Cyclomatic complexity gate for the frontend JS/JSX codebase.
 * Uses @babel/parser (transitive dep of @vitejs/plugin-react — no extra install needed).
 *
 * Thresholds:
 *   COMPLEX_MIN     — functions at or above this count toward the budget
 *   CRITICAL_MIN    — zero-tolerance list
 *   MAX_COMPLEX_PCT — max allowed share of complex functions (0-100)
 *   MAX_CRITICAL    — max allowed count of critical functions
 *
 * Run:
 *   node frontend/test_complexity.js
 *   node frontend/test_complexity.js --complex-min 8 --critical-min 15 --max-pct 5 --max-critical 0
 */

import { readFileSync, readdirSync, statSync } from "fs";
import { join, relative, extname } from "path";
import { fileURLToPath } from "url";
import { parse } from "@babel/parser";

// -- defaults ----------------------------------------------------------------
const COMPLEX_MIN = 10;
const CRITICAL_MIN = 40;
const MAX_COMPLEX_PCT = 5;
const MAX_CRITICAL = 0;

const __dirname = fileURLToPath(new URL(".", import.meta.url));
const SRC_DIR = join(__dirname, "renderer", "src");

// -- AST helpers -------------------------------------------------------------

const BRANCH_NODES = new Set([
  "IfStatement",
  "ConditionalExpression",
  "ForStatement",
  "ForInStatement",
  "ForOfStatement",
  "WhileStatement",
  "DoWhileStatement",
  "CatchClause",
]);

function isBranch(node) {
  if (BRANCH_NODES.has(node.type)) return true;
  if (node.type === "SwitchCase" && node.test !== null) return true;
  if (
    node.type === "LogicalExpression" &&
    (node.operator === "&&" || node.operator === "||" || node.operator === "??")
  )
    return true;
  return false;
}

function isFn(node) {
  return (
    node.type === "FunctionDeclaration" ||
    node.type === "FunctionExpression" ||
    node.type === "ArrowFunctionExpression" ||
    node.type === "ObjectMethod" ||
    node.type === "ClassMethod" ||
    node.type === "ClassPrivateMethod"
  );
}

function fnName(node, parent) {
  if (node.key?.name) return node.key.name;
  if (node.id?.name) return node.id.name;
  if (parent?.type === "VariableDeclarator" && parent.id?.name)
    return parent.id.name;
  if (
    parent?.type === "AssignmentExpression" ||
    parent?.type === "AssignmentPattern"
  ) {
    const l = parent.left;
    return l?.name || l?.property?.name || "<anonymous>";
  }
  if (parent?.type === "Property" || parent?.type === "ObjectProperty")
    return parent.key?.name || parent.key?.value || "<anonymous>";
  return "<anonymous>";
}

// Walk the AST; fnStack tracks open function scopes (innermost last).
function walk(node, parent, records, filePath, fnStack) {
  if (!node || typeof node !== "object") return;

  if (isFn(node)) {
    const entry = { name: fnName(node, parent), cc: 1, line: node.loc?.start?.line ?? 0 };
    fnStack.push(entry);
    walkChildren(node, records, filePath, fnStack);
    fnStack.pop();
    records.push({ file: filePath, name: entry.name, cc: entry.cc, line: entry.line });
    return;
  }

  if (fnStack.length > 0 && isBranch(node)) {
    fnStack[fnStack.length - 1].cc += 1;
  }

  walkChildren(node, records, filePath, fnStack);
}

function walkChildren(node, records, filePath, fnStack) {
  for (const key of Object.keys(node)) {
    const child = node[key];
    if (Array.isArray(child)) {
      for (const item of child)
        if (item && typeof item === "object" && item.type)
          walk(item, node, records, filePath, fnStack);
    } else if (child && typeof child === "object" && child.type) {
      walk(child, node, records, filePath, fnStack);
    }
  }
}

// -- file scanning -----------------------------------------------------------

function collectFile(absPath, srcDir) {
  const code = readFileSync(absPath, "utf-8");
  let ast;
  try {
    ast = parse(code, {
      sourceType: "module",
      plugins: [
        "jsx",
        "classProperties",
        "optionalChaining",
        "nullishCoalescingOperator",
        "objectRestSpread",
        "asyncGenerators",
      ],
    });
  } catch (e) {
    process.stderr.write(
      `  [skip] ${relative(srcDir, absPath)}: ${e.message}\n`
    );
    return [];
  }
  const records = [];
  const rel = relative(srcDir, absPath).replace(/\\/g, "/");
  walk(ast.program, null, records, rel, []);
  return records;
}

function* walkDir(dir) {
  for (const entry of readdirSync(dir).sort()) {
    const full = join(dir, entry);
    if (statSync(full).isDirectory()) {
      yield* walkDir(full);
    } else {
      const ext = extname(entry).toLowerCase();
      if (ext === ".js" || ext === ".jsx") yield full;
    }
  }
}

function collect(srcDir) {
  const records = [];
  for (const file of walkDir(srcDir)) records.push(...collectFile(file, srcDir));
  return records;
}

// -- reporting ---------------------------------------------------------------

function grade(cc) {
  if (cc <= 5) return "A";
  if (cc <= 10) return "B";
  if (cc <= 15) return "C";
  if (cc <= 20) return "D";
  if (cc <= 25) return "E";
  return "F";
}

function run({ complexMin, criticalMin, maxComplexPct, maxCritical }) {
  const records = collect(SRC_DIR);
  const total = records.length;
  if (total === 0) {
    console.log("No JS/JSX functions found.");
    return true;
  }

  const complexUnits = records.filter((r) => r.cc >= complexMin);
  const criticalUnits = records.filter((r) => r.cc >= criticalMin);
  const complexPct = (complexUnits.length / total) * 100;

  const sep = "-".repeat(60);
  console.log(`\n${sep}`);
  console.log(`  Total units analysed : ${total}`);
  console.log(
    `  Complex  (cc>=${String(complexMin).padStart(2)})      : ${String(complexUnits.length).padStart(4)}  (${complexPct.toFixed(1)} %)`
  );
  console.log(
    `  Critical (cc>=${String(criticalMin).padStart(2)})      : ${String(criticalUnits.length).padStart(4)}`
  );
  console.log(sep);

  if (complexUnits.length) {
    console.log(`\n  Complex units (cc>=${complexMin}):`);
    for (const r of complexUnits.sort((a, b) => b.cc - a.cc))
      console.log(
        `    [${grade(r.cc)}:${String(r.cc).padStart(2)}]  ${r.file}:${r.line}  ${r.name}`
      );
  }

  const failures = [];
  if (complexPct > maxComplexPct)
    failures.push(
      `Complex units: ${complexPct.toFixed(1)} % > allowed ${maxComplexPct} %`
    );
  if (criticalUnits.length > maxCritical) {
    failures.push(
      `Critical units: ${criticalUnits.length} > allowed ${maxCritical}`
    );
    for (const r of criticalUnits.sort((a, b) => b.cc - a.cc))
      failures.push(
        `  >> [${grade(r.cc)}:${r.cc}]  ${r.file}:${r.line}  ${r.name}`
      );
  }

  console.log();
  if (failures.length) {
    console.log("  FAIL");
    for (const msg of failures) console.log(`  ${msg}`);
    console.log();
    return false;
  }
  console.log(
    `  PASS  (complex ${complexPct.toFixed(1)} % <= ${maxComplexPct} %, critical ${criticalUnits.length} <= ${maxCritical})`
  );
  console.log();
  return true;
}

// -- CLI ---------------------------------------------------------------------

function parseArgs() {
  const args = process.argv.slice(2);
  const get = (flag, def) => {
    const i = args.indexOf(flag);
    return i !== -1 && args[i + 1] != null ? args[i + 1] : def;
  };
  return {
    complexMin: parseInt(get("--complex-min", COMPLEX_MIN), 10),
    criticalMin: parseInt(get("--critical-min", CRITICAL_MIN), 10),
    maxComplexPct: parseFloat(get("--max-pct", MAX_COMPLEX_PCT)),
    maxCritical: parseInt(get("--max-critical", MAX_CRITICAL), 10),
  };
}

const ok = run(parseArgs());
process.exit(ok ? 0 : 1);
