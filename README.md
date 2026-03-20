# AuthKeyHub

Secure authentication system built with FastAPI, JWT tokens, and PostgreSQL.

## Features

- User registration with validation and role assignment
- JWT authentication (access + refresh tokens)
- Role-based user model (`researcher`, `government`, `student`)
- Protected routes with token verification
- Password hashing with bcrypt
- PostgreSQL database (SQLite for local development)
- Docker & Docker Compose support
- CI with GitHub Actions (flake8, isort, pylint, mypy)

## Quick Start (Docker)

```bash
docker-compose up --build
```

App: http://localhost:8000
Docs: http://localhost:8000/docs

## Local Development

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `DATABASE_URL` | `sqlite:///./auth.db` | Database connection string |
| `SECRET_KEY` | `your-super-secret-key-...` | JWT signing key (change in production!) |
| `ALGORITHM` | `HS256` | JWT algorithm |
| `ACCESS_TOKEN_EXPIRE_MINUTES` | `30` | Access token TTL |
| `REFRESH_TOKEN_EXPIRE_DAYS` | `7` | Refresh token TTL |
| `APP_NAME` | `AuthKeyHub` | Application name |
| `DEBUG` | `False` | Debug mode |

## API Endpoints

### General

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | App info |
| `GET` | `/health` | Health check |
| `GET` | `/docs` | Swagger UI |

### Authentication (`/api/jacobs/auth`)

| Method | Path | Description |
|---|---|---|
| `POST` | `/register` | Register a new user |
| `POST` | `/login` | Login and get tokens |
| `POST` | `/refresh` | Refresh access token |
| `GET` | `/me` | Get current user info |
| `POST` | `/logout` | Logout |
| `GET` | `/status` | Auth system status |

### Register

```bash
curl -X POST http://localhost:8000/api/jacobs/auth/register \
  -H "Content-Type: application/json" \
  -d '{"email": "user@example.com", "username": "user", "password": "pass123", "role": "researcher"}'
```

Response:
```json
{
  "id": 1,
  "email": "user@example.com",
  "username": "user",
  "role": "researcher",
  "is_active": true,
  "created_at": "2025-01-01T12:00:00",
  "updated_at": null
}
```

### Login

```bash
curl -X POST http://localhost:8000/api/jacobs/auth/login \
  -H "Content-Type: application/json" \
  -d '{"email": "user@example.com", "password": "pass123"}'
```

### Get Current User

```bash
curl http://localhost:8000/api/jacobs/auth/me \
  -H "Authorization: Bearer <access_token>"
```

## User Roles

| Role | Value |
|---|---|
| Researcher | `researcher` |
| Government employee | `government` |
| Student | `student` |

Default role on registration: `student`.

## Token System

- **Access token** — short-lived (30 min), used for API access
- **Refresh token** — long-lived (7 days), used to renew access tokens

## Linting

```bash
flake8 --config=config/setup.cfg main.py backend/
isort --settings-path=config/pyproject.toml --check-only main.py backend/
pylint --rcfile=config/.pylintrc main.py backend/
mypy --config-file=config/pyproject.toml main.py backend/
```

## Project Structure

```
AuthKeyHub/
├── backend/
│   ├── __init__.py
│   ├── auth.py            # Auth routes and dependencies
│   ├── config.py           # App settings
│   ├── database.py         # SQLAlchemy models
│   ├── schemas.py          # Pydantic schemas
│   └── security.py         # JWT and password utils
├── config/
│   ├── pyproject.toml      # isort, pylint, mypy config
│   ├── setup.cfg           # flake8 config
│   └── .pylintrc           # pylint threshold
├── docker/
│   └── Dockerfile          # Container build
├── .github/workflows/
│   └── lint.yml            # CI pipeline
├── main.py                 # FastAPI app entry point
├── docker-compose.yml
├── requirements.txt
└── README.md
```
