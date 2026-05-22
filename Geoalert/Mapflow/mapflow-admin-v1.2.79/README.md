# Development procedures

## Prerequisites

1. Node Version Manager (https://github.com/nvm-sh/nvm)

## After checkout

1. Install nodejs 16.x.X `nvm i 16`
2. Install React Developer Tools browser Add-on
3. Run `npm i` to install dependencies
4. Compile lungui messages, run `npm run compile`

## External depedencies

1. WM Client requires access to WM GraphQL API. In production environment Backend is deployed to http://backend:8080
   and available via /graphql and /rest endpoints. See: `whitemaps.nginx.conf` for deployment details. This URL is configured by `window.location.origin/graphql`
   environment variable. For development environment this variable is set to `https://whitemaps-duty.mapflow.ai/graphql`

2. WM Client REST API

## Running locally

Start application in development node `npm start`. In this mode WM client will use https://duty-whitemaps.mapflow.ai/graphql
as a backend endpoint.

## I18N

After adding new `<Trans>...</Trans>` constant you need to provide a translation. Run 'npm run extract' then add translation
to the `src/locales/ru/messages.po` file. Then run 'npm run compile' to compile `messages.po` to `messages.js`.

## Tests

Run `npm run test` to run tests. Note: tests are broken for now and needs to be fixed.

## Production deployment

See: `Dockerfile` and `.gitlab-ci.yml`
