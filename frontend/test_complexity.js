/**
 * Cyclomatic complexity gate for renderer source JS/JSX. Electron is linted
 * and dependency-checked separately by npm run quality.
 * Uses @babel/parser, supplied by @vitejs/plugin-react.
 *
 * Thresholds:
 *   COMPLEX_MIN     вЂ” functions at or above this count toward the budget
 *   CRITICAL_MIN    вЂ” zero-tolerance list
 *   MAX_COMPLEX_PCT вЂ” max allowed share of complex functions (0-100)
 *   MAX_CRITICAL    вЂ” max allowed count of critical functions
 *
 * Run:
 *   node frontend/test_complexity.js
 *   node frontend/test_complexity.js --complex-min 8 --critical-min 15
 *     --max-pct 5 --max-critical 0
 */

import { readFileSync, readdirSync, statSync } from 'fs';
import { join, relative, extname } from 'path';
import { fileURLToPath } from 'url';
import { parse } from '@babel/parser';

// -- defaults ----------------------------------------------------------------
const COMPLEX_MIN = 10;
const CRITICAL_MIN = 21;
const MAX_COMPLEX_PCT = 5;
const MAX_CRITICAL = 0;

const __dirname = fileURLToPath(new URL('.', import.meta.url));
const SRC_DIR = join(__dirname, 'renderer', 'src');

// -- AST helpers -------------------------------------------------------------

const BRANCH_NODES = new Set([
  'IfStatement',
  'ConditionalExpression',
  'ForStatement',
  'ForInStatement',
  'ForOfStatement',
  'WhileStatement',
  'DoWhileStatement',
  'CatchClause',
]);

const isBranch = (node) => {
  if (BRANCH_NODES.has(node.type)) return true;
  if (node.type === 'SwitchCase' && node.test !== null) return true;
  if (
    node.type === 'LogicalExpression' &&
    (node.operator === '&&' || node.operator === '||' || node.operator === '??')
  ) {
    return true;
  }
  return false;
};

const isFunction = (node) =>
  node.type === 'FunctionDeclaration' ||
  node.type === 'FunctionExpression' ||
  node.type === 'ArrowFunctionExpression' ||
  node.type === 'ObjectMethod' ||
  node.type === 'ClassMethod' ||
  node.type === 'ClassPrivateMethod';

const functionName = (node, parent) => {
  if (node.key?.name) return node.key.name;
  if (node.id?.name) return node.id.name;
  if (parent?.type === 'VariableDeclarator' && parent.id?.name) {
    return parent.id.name;
  }
  if (
    parent?.type === 'AssignmentExpression' ||
    parent?.type === 'AssignmentPattern'
  ) {
    const left = parent.left;
    return left?.name || left?.property?.name || '<anonymous>';
  }
  if (parent?.type === 'Property' || parent?.type === 'ObjectProperty') {
    return parent.key?.name || parent.key?.value || '<anonymous>';
  }
  return '<anonymous>';
};

const walkChildren = (node, records, filePath, functionStack, visit) => {
  for (const key of Object.keys(node)) {
    const child = node[key];
    if (Array.isArray(child)) {
      for (const item of child) {
        if (item && typeof item === 'object' && item.type) {
          visit(item, node, records, filePath, functionStack);
        }
      }
    } else if (child && typeof child === 'object' && child.type) {
      visit(child, node, records, filePath, functionStack);
    }
  }
};

// Walk the AST; functionStack tracks open function scopes (innermost last).
const walk = (node, parent, records, filePath, functionStack) => {
  if (!node || typeof node !== 'object') return;

  if (isFunction(node)) {
    const entry = {
      name: functionName(node, parent),
      cc: 1,
      line: node.loc?.start?.line ?? 0,
    };
    functionStack.push(entry);
    walkChildren(node, records, filePath, functionStack, walk);
    functionStack.pop();
    records.push({
      file: filePath,
      name: entry.name,
      cc: entry.cc,
      line: entry.line,
    });
    return;
  }

  if (functionStack.length > 0 && isBranch(node)) {
    functionStack[functionStack.length - 1].cc += 1;
  }

  walkChildren(node, records, filePath, functionStack, walk);
};

// -- file scanning -----------------------------------------------------------

const collectFile = (absolutePath, sourceDirectory) => {
  const code = readFileSync(absolutePath, 'utf-8');
  let ast;
  try {
    ast = parse(code, {
      sourceType: 'module',
      plugins: [
        'jsx',
        'classProperties',
        'optionalChaining',
        'nullishCoalescingOperator',
        'objectRestSpread',
        'asyncGenerators',
      ],
    });
  } catch (error) {
    process.stderr.write(
      `  [skip] ${relative(sourceDirectory, absolutePath)}: ${error.message}\n`,
    );
    throw new Error(
      `Cannot analyze ${relative(sourceDirectory, absolutePath)}`,
      {
        cause: error,
      },
    );
  }
  const records = [];
  const relativePath = relative(sourceDirectory, absolutePath).replace(
    /\\/g,
    '/',
  );
  walk(ast.program, null, records, relativePath, []);
  return records;
};

const walkDir = function* (directory) {
  for (const entry of readdirSync(directory).sort()) {
    const full = join(directory, entry);
    if (statSync(full).isDirectory()) {
      yield* walkDir(full);
    } else {
      const extension = extname(entry).toLowerCase();
      if (extension === '.js' || extension === '.jsx') yield full;
    }
  }
};

const collect = (sourceDirectory) => {
  const records = [];
  for (const file of walkDir(sourceDirectory)) {
    records.push(...collectFile(file, sourceDirectory));
  }
  return records;
};

// -- reporting ---------------------------------------------------------------

const grade = (cc) => {
  if (cc <= 5) return 'A';
  if (cc <= 10) return 'B';
  if (cc <= 15) return 'C';
  if (cc <= 20) return 'D';
  if (cc <= 25) return 'E';
  return 'F';
};

// -- CLI ---------------------------------------------------------------------

const parseArgs = () => {
  const args = process.argv.slice(2);
  const get = (flag, fallback) => {
    const index = args.indexOf(flag);
    return index !== -1 &&
      args[index + 1] !== null &&
      args[index + 1] !== undefined
      ? args[index + 1]
      : fallback;
  };
  const limits = {
    complexMin: Number(get('--complex-min', COMPLEX_MIN)),
    criticalMin: Number(get('--critical-min', CRITICAL_MIN)),
    maxComplexPct: Number(get('--max-pct', MAX_COMPLEX_PCT)),
    maxCritical: Number(get('--max-critical', MAX_CRITICAL)),
  };
  if (!Object.values(limits).every(Number.isFinite)) {
    throw new Error('Complexity limits must be finite numbers');
  }
  if (
    !Number.isInteger(limits.complexMin) ||
    limits.complexMin < 1 ||
    !Number.isInteger(limits.criticalMin) ||
    limits.criticalMin < limits.complexMin ||
    !Number.isInteger(limits.maxCritical) ||
    limits.maxCritical < 0 ||
    limits.maxComplexPct < 0 ||
    limits.maxComplexPct > 100
  ) {
    throw new Error('Invalid complexity limits');
  }
  return limits;
};

const complexUnitLabel = (record) => {
  const complexity = `${grade(record.cc)}:${String(record.cc).padStart(2)}`;
  return `    [${complexity}]  ${record.file}:${record.line}  ${record.name}`;
};

const run = ({ complexMin, criticalMin, maxComplexPct, maxCritical }) => {
  const records = collect(SRC_DIR);
  const total = records.length;
  if (total === 0) {
    console.log('No JS/JSX functions found.');
    return true;
  }

  const complexUnits = records.filter((r) => r.cc >= complexMin);
  const criticalUnits = records.filter((r) => r.cc >= criticalMin);
  const complexPct = (complexUnits.length / total) * 100;

  const separator = '-'.repeat(60);
  const complexLabel = String(complexMin).padStart(2);
  const complexCount = String(complexUnits.length).padStart(4);
  const criticalLabel = String(criticalMin).padStart(2);
  const criticalCount = String(criticalUnits.length).padStart(4);
  console.log(`\n${separator}`);
  console.log(`  Total units analysed : ${total}`);
  console.log(
    [
      `  Complex  (cc>=${complexLabel})      : ${complexCount}`,
      `  (${complexPct.toFixed(1)} %)`,
    ].join(''),
  );
  console.log(`  Critical (cc>=${criticalLabel})      : ${criticalCount}`);
  console.log(separator);

  if (complexUnits.length) {
    console.log(`\n  Complex units (cc>=${complexMin}):`);
    for (const r of complexUnits.sort((a, b) => b.cc - a.cc)) {
      console.log(complexUnitLabel(r));
    }
  }

  const failures = [];
  if (complexPct > maxComplexPct) {
    failures.push(
      `Complex units: ${complexPct.toFixed(1)} % > allowed ${maxComplexPct} %`,
    );
  }
  if (criticalUnits.length > maxCritical) {
    failures.push(
      `Critical units: ${criticalUnits.length} > allowed ${maxCritical}`,
    );
    for (const r of criticalUnits.sort((a, b) => b.cc - a.cc)) {
      failures.push(
        `  >> [${grade(r.cc)}:${r.cc}]  ${r.file}:${r.line}  ${r.name}`,
      );
    }
  }

  console.log();
  if (failures.length) {
    console.log('  FAIL');
    for (const message of failures) console.log(`  ${message}`);
    console.log();
    return false;
  }
  console.log(
    [
      '  PASS  (complex',
      `${complexPct.toFixed(1)} % <= ${maxComplexPct} %,`,
      `critical ${criticalUnits.length} <= ${maxCritical})`,
    ].join(' '),
  );
  console.log();
  return true;
};

const ok = run(parseArgs());
process.exit(ok ? 0 : 1);
