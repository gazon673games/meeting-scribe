import metarhia from 'eslint-config-metarhia';

const files = [
  'frontend/**/*.{js,jsx,cjs}',
  '*.config.js',
  '.dependency-cruiser.cjs',
  'eslint.config.js',
];

const jsxReferences = {
  meta: { schema: [] },
  create: (context) => ({
    JSXOpeningElement: (node) => {
      let name = node.name;
      const isMember = name.type === 'JSXMemberExpression';
      while (name.type === 'JSXMemberExpression') name = name.object;
      if (
        name.type === 'JSXIdentifier' &&
        (isMember || /^[A-Z]/u.test(name.name))
      ) {
        context.sourceCode.markVariableAsUsed(name.name, node);
      }
    },
  }),
};

export default [
  ...metarhia.map((config) => ({ ...config, files })),
  {
    files: ['**/*.{js,jsx}'],
    languageOptions: {
      sourceType: 'module',
      parserOptions: { ecmaFeatures: { jsx: true } },
    },
  },
  {
    files: ['frontend/renderer/**/*.{js,jsx}'],
    languageOptions: {
      globals: {
        window: 'readonly',
        document: 'readonly',
        navigator: 'readonly',
        HTMLElement: 'readonly',
        localStorage: 'readonly',
        ResizeObserver: 'readonly',
        requestAnimationFrame: 'readonly',
        cancelAnimationFrame: 'readonly',
        getComputedStyle: 'readonly',
      },
    },
    plugins: { jsx: { rules: { references: jsxReferences } } },
    rules: {
      'jsx/references': 'error',
      // Effects may return a cleanup function or an explicit undefined.
      'consistent-return': ['error', { treatUndefinedAsUnspecified: false }],
    },
  },
  {
    files,
    rules: {
      camelcase: ['error', { properties: 'never' }],
      'func-style': ['error', 'expression'],
      'no-restricted-syntax': [
        'error',
        {
          selector:
            'FunctionExpression[generator=false]' +
            ':not(MethodDefinition > FunctionExpression)' +
            ':not(Property[method=true] > FunctionExpression)' +
            ':not(AssignmentExpression[left.type=MemberExpression]' +
            ' > FunctionExpression)',
          message: 'Use an arrow function for a context-free function.',
        },
      ],
      'no-extra-parens': 'off',
      'max-len': [
        'error',
        {
          code: 80,
          ignoreUrls: true,
          ignoreRegExpLiterals: true,
          ignorePattern: '^\\s*(?:import\\s|test\\()',
        },
      ],
    },
  },
];
