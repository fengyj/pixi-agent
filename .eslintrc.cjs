module.exports = {
  root: true,
  parser: '@typescript-eslint/parser',
  parserOptions: {
    ecmaVersion: 2020,
    sourceType: 'module'
  },
  plugins: ['@typescript-eslint', 'import-x'],
  extends: ['eslint:recommended', 'plugin:@typescript-eslint/recommended', 'plugin:import-x/recommended', 'prettier'],
  env: {
    node: true,
    es2020: true,
    jest: true
  },
  rules: {
    'import-x/no-unresolved': 'off'
  }
};
