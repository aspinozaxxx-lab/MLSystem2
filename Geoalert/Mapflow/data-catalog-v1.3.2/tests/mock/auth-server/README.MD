# Mock server for handling authorization requests from data-catalog app

### Run:

`python3 auth_server.py`

Server handles the authorization requests coming from data-catalog app and
responses accordingly 

### Send authorization requests to: 

`localhost:8050/auth/`

### Sample credentials:
`"some_login": "1234567890"
`

`"another_login": "another_password"`

Sample responses:
- 
- if authorisation is ok:
`{"login":"login@login.com", "isAdmin":"False", "spaceLimit":992976.992636}`
- if authorisation header is not present:
`{"code":"ACCESS_DENIED", "message":"Access denied."}
`
- if wrong credentials or user doesn't exist in database:
`{"code":"AUTHENTICATION_ERROR", "message":"Wrong login/password."}
`

