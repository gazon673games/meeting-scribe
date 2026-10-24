'use strict';

module.exports = {
  forbidden: [
    {
      name: 'no-circular',
      severity: 'error',
      comment:
        'Circular dependencies make module boundaries harder to reason about.',
      from: {},
      to: { circular: true },
    },
    {
      name: 'no-unresolvable',
      severity: 'error',
      comment: 'All frontend and Electron imports should resolve.',
      from: {},
      to: { couldNotResolve: true },
    },
    {
      name: 'renderer-not-to-electron',
      severity: 'error',
      comment: 'Renderer code should access Electron through the preload API.',
      from: { path: '^frontend/renderer/src' },
      to: { path: '^frontend/electron' },
    },
    {
      name: 'electron-not-to-renderer-src',
      severity: 'error',
      comment: 'Electron should not import React renderer implementation.',
      from: { path: '^frontend/electron' },
      to: { path: '^frontend/renderer/src' },
    },
  ],
  options: {
    doNotFollow: {
      path: 'node_modules|coverage|dist|build',
    },
    exclude: {
      path: 'node_modules|coverage|dist|build',
    },
    enhancedResolveOptions: {
      extensions: ['.js', '.jsx', '.cjs', '.mjs', '.json'],
      mainFields: ['module', 'main'],
      conditionNames: ['import', 'require', 'node', 'default'],
    },
    reporterOptions: {
      dot: {
        collapsePattern: 'node_modules/[^/]+',
      },
      archi: {
        collapsePattern: '^(frontend/(electron|renderer/src/[^/]+)).*',
      },
    },
  },
};
