# BEACON API Documentation

## Overview

The BEACON API provides endpoints for cancer diagnosis and treatment support. It follows RESTful principles and uses JSON for request/response payloads.

Base URL: `https://api.beacon.example.com`

## Authentication

### Get Access Token

```http
POST /api/v1/auth/login
```

Request body:
```json
{
    "username": "string",
    "password": "string"
}
```

Response:
```json
{
    "access_token": "string",
    "token_type": "bearer"
}
```

All subsequent requests must include the access token in the Authorization header:
```http
Authorization: Bearer <access_token>
```

## Clinical Data API

### Upload Clinical Data

```http
POST /api/v1/clinical/upload
Content-Type: multipart/form-data
```

Request body:
- `file`: CSV file containing clinical data

Required columns:
- `age`: numeric
- `weight`: numeric
- `height`: numeric
- `blood_pressure_systolic`: numeric
- `blood_pressure_diastolic`: numeric
- `heart_rate`: numeric
- `temperature`: numeric
- `glucose`: numeric
- `cholesterol`: numeric
- `smoking`: string (Never/Former/Current)

Response:
```json
{
    "message": "File uploaded successfully",
    "filename": "string"
}
```

### Process Clinical Data

```http
POST /api/v1/clinical/process
```

Request body:
```json
{
    "filename": "string"
}
```

Response:
```json
{
    "message": "Data processed successfully"
}
```

### Get Clinical Predictions

```http
POST /api/v1/clinical/predict
Content-Type: multipart/form-data
```

Request body:
- `file`: CSV file containing clinical data

Response:
```json
[
    {
        "diabetes_prob": 0.123,
        "hypertension_prob": 0.456
    }
]
```

## Imaging Data API

### Upload Imaging Data

```http
POST /api/v1/imaging/upload
Content-Type: multipart/form-data
```

Request body:
- `file`: NPY file containing imaging data (shape: [samples, channels, height, width])

Response:
```json
{
    "message": "File uploaded successfully",
    "filename": "string"
}
```

### Upload Image Labels

```http
POST /api/v1/imaging/upload-labels
Content-Type: multipart/form-data
```

Request body:
- `file`: NPY file containing image labels

Response:
```json
{
    "message": "Labels uploaded successfully",
    "filename": "string"
}
```

### Process Imaging Data

```http
POST /api/v1/imaging/process
```

Request body:
```json
{
    "data_filename": "string",
    "labels_filename": "string"
}
```

Response:
```json
{
    "message": "Data processed successfully"
}
```

### Get Imaging Predictions

```http
POST /api/v1/imaging/predict
Content-Type: multipart/form-data
```

Request body:
- `file`: NPY file containing imaging data

Response:
```json
[
    {
        "abnormality_prob": 0.789
    }
]
```

## Genomic Data API

### Upload Expression Data

```http
POST /api/v1/genomic/upload-expression
Content-Type: multipart/form-data
```

Request body:
- `file`: NPZ file containing expression data

Response:
```json
{
    "message": "Expression data uploaded successfully",
    "filename": "string"
}
```

### Upload Mutation Data

```http
POST /api/v1/genomic/upload-mutations
Content-Type: multipart/form-data
```

Request body:
- `file`: NPZ file containing mutation data

Response:
```json
{
    "message": "Mutation data uploaded successfully",
    "filename": "string"
}
```

### Upload CNV Data

```http
POST /api/v1/genomic/upload-cnv
Content-Type: multipart/form-data
```

Request body:
- `file`: NPZ file containing CNV data

Response:
```json
{
    "message": "CNV data uploaded successfully",
    "filename": "string"
}
```

### Process Genomic Data

```http
POST /api/v1/genomic/process
```

Request body:
```json
{
    "expression_filename": "string",
    "mutations_filename": "string",
    "cnv_filename": "string"
}
```

Response:
```json
{
    "message": "Data processed successfully"
}
```

### Get Genomic Predictions

```http
POST /api/v1/genomic/predict
Content-Type: multipart/form-data
```

Request body:
- `expression_file`: NPZ file containing expression data
- `mutations_file`: NPZ file containing mutation data
- `cnv_file`: NPZ file containing CNV data

Response:
```json
[
    {
        "expression_high_prob": 0.345
    }
]
```

## Survival Data API

### Upload Survival Data

```http
POST /api/v1/survival/upload
Content-Type: multipart/form-data
```

Request body:
- `file`: CSV file containing survival data

Required columns:
- `time`: numeric (days)
- `event`: binary (0/1)
- `age`: numeric
- `stage`: string (I/II/III/IV)
- `grade`: string (Low/Medium/High)

Response:
```json
{
    "message": "File uploaded successfully",
    "filename": "string"
}
```

### Process Survival Data

```http
POST /api/v1/survival/process
```

Request body:
```json
{
    "filename": "string"
}
```

Response:
```json
{
    "message": "Data processed successfully"
}
```

### Get Survival Predictions

```http
POST /api/v1/survival/predict
Content-Type: multipart/form-data
```

Request body:
- `file`: CSV file containing survival data

Response:
```json
[
    {
        "predicted_time": 365.0,
        "event_prob": 0.567
    }
]
```

## Error Responses

### 400 Bad Request
```json
{
    "detail": "Invalid data format"
}
```

### 401 Unauthorized
```json
{
    "detail": "Could not validate credentials"
}
```

### 403 Forbidden
```json
{
    "detail": "Not enough permissions"
}
```

### 404 Not Found
```json
{
    "detail": "Resource not found"
}
```

### 500 Internal Server Error
```json
{
    "detail": "Internal server error"
}
```

## Rate Limiting

API endpoints are rate limited to:
- 100 requests per minute for authenticated users
- 10 requests per minute for unauthenticated users

Rate limit headers:
```http
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1620000000
```

## Pagination

List endpoints support pagination using query parameters:
- `skip`: Number of items to skip
- `limit`: Maximum number of items to return

Example:
```http
GET /api/v1/resource?skip=0&limit=10
```

Response includes pagination metadata:
```json
{
    "items": [],
    "total": 100,
    "skip": 0,
    "limit": 10
}
```

## Versioning

The API is versioned using URL path versioning:
- Current version: `v1`
- Base path: `/api/v1`

## CORS

The API supports Cross-Origin Resource Sharing (CORS) for specified origins.

Allowed origins:
- `http://localhost:3000`
- `http://localhost:8000`
- Additional origins can be configured in settings

## WebSocket Support

Real-time updates are available through WebSocket connections:

```javascript
const ws = new WebSocket('wss://api.beacon.example.com/ws');

ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    console.log(data);
};
```

## API Clients

Example API client usage:

### Python
```python
import requests

class BeaconClient:
    def __init__(self, base_url, token):
        self.base_url = base_url
        self.headers = {'Authorization': f'Bearer {token}'}
    
    def predict_clinical(self, file_path):
        with open(file_path, 'rb') as f:
            files = {'file': f}
            response = requests.post(
                f'{self.base_url}/api/v1/clinical/predict',
                headers=self.headers,
                files=files
            )
        return response.json()
```

### JavaScript
```javascript
class BeaconClient {
    constructor(baseUrl, token) {
        this.baseUrl = baseUrl;
        this.headers = {
            'Authorization': `Bearer ${token}`
        };
    }

    async predictClinical(file) {
        const formData = new FormData();
        formData.append('file', file);

        const response = await fetch(
            `${this.baseUrl}/api/v1/clinical/predict`,
            {
                method: 'POST',
                headers: this.headers,
                body: formData
            }
        );
        return response.json();
    }
}
```
