# Data Analyst Agent API - Deployment Guide

This guide covers multiple deployment options for the Data Analyst Agent API.

## 🚀 Quick Start

### Local Development
```bash
# Install dependencies
pip install -r requirements.txt

# Run the API
python run.py

# Test the API
python test_api.py
```

### Using the Deployment Script
```bash
python deploy.py
```

## 🌐 Cloud Deployment Options

### 1. Railway (Recommended - Free Tier Available)

Railway is the easiest option with automatic deployments from GitHub.

1. **Connect to GitHub:**
   - Go to [railway.app](https://railway.app)
   - Sign in with GitHub
   - Click "New Project" → "Deploy from GitHub repo"

2. **Configure Environment:**
   - Railway will automatically detect the Dockerfile
   - No additional configuration needed

3. **Get Your URL:**
   - Railway will provide a URL like: `https://your-app-name.railway.app`
   - Your API endpoint will be: `https://your-app-name.railway.app/api/`

### 2. Render (Free Tier Available)

1. **Create Account:**
   - Go to [render.com](https://render.com)
   - Sign up with GitHub

2. **Deploy:**
   - Click "New" → "Web Service"
   - Connect your GitHub repository
   - Set build command: `pip install -r requirements.txt`
   - Set start command: `uvicorn api.main:app --host=0.0.0.0 --port=$PORT`

3. **Environment Variables:**
   - No special environment variables needed

### 3. Heroku

1. **Install Heroku CLI:**
   ```bash
   # macOS
   brew install heroku/brew/heroku
   
   # Windows
   # Download from https://devcenter.heroku.com/articles/heroku-cli
   ```

2. **Create Procfile:**
   ```bash
   python deploy.py
   # Select option 6 to create Procfile
   ```

3. **Deploy:**
   ```bash
   heroku create your-app-name
   git add .
   git commit -m "Deploy to Heroku"
   git push heroku main
   ```

### 4. Vercel

1. **Install Vercel CLI:**
   ```bash
   npm install -g vercel
   ```

2. **Create Configuration:**
   ```bash
   python deploy.py
   # Select option 5 to create vercel.json
   ```

3. **Deploy:**
   ```bash
   vercel --prod
   ```

### 5. Google Cloud Run

1. **Install Google Cloud CLI:**
   ```bash
   # Follow instructions at: https://cloud.google.com/sdk/docs/install
   ```

2. **Build and Deploy:**
   ```bash
   # Build the Docker image
   docker build -t gcr.io/YOUR_PROJECT_ID/data-analyst-api .
   
   # Push to Google Container Registry
   docker push gcr.io/YOUR_PROJECT_ID/data-analyst-api
   
   # Deploy to Cloud Run
   gcloud run deploy data-analyst-api \
     --image gcr.io/YOUR_PROJECT_ID/data-analyst-api \
     --platform managed \
     --region us-central1 \
     --allow-unauthenticated
   ```

## 🐳 Docker Deployment

### Build and Run Locally
```bash
# Build the image
docker build -t data-analyst-api .

# Run the container
docker run -p 8000:8000 data-analyst-api
```

### Deploy to Docker Hub
```bash
# Tag your image
docker tag data-analyst-api your-username/data-analyst-api

# Push to Docker Hub
docker push your-username/data-analyst-api
```

## 🧪 Testing Your Deployment

### Test with curl
```bash
# Create a test question file
echo "Scrape the list of highest grossing films from Wikipedia. It is at the URL:
https://en.wikipedia.org/wiki/List_of_highest-grossing_films

Answer the following questions and respond with a JSON array of strings containing the answer.

1. How many \$2 bn movies were released before 2000?
2. Which is the earliest film that grossed over \$1.5 bn?
3. What's the correlation between the Rank and Peak?
4. Draw a scatterplot of Rank and Peak along with a dotted red regression line through it.
   Return as a base-64 encoded data URI, \`\"data:image/png;base64,iVBORw0KG...\"\` under 100,000 bytes." > test_question.txt

# Test the API
curl "https://your-app-url.com/api/" -F "questions_txt=@test_question.txt"
```

### Test with Python
```bash
python test_api.py
```

## 📋 API Specification

### Endpoint
- **URL:** `POST https://your-app-url.com/api/`
- **Timeout:** 3 minutes maximum

### Request Format
```bash
curl "https://your-app-url.com/api/" \
  -F "questions_txt=@questions.txt" \
  -F "image.png=@image.png" \
  -F "data.csv=@data.csv"
```

### Response Format
```json
[1, "Titanic", 0.485782, "data:image/png;base64,iVBORw0KG..."]
```

- **Element 1:** Count (integer)
- **Element 2:** Movie name (string)
- **Element 3:** Correlation (float)
- **Element 4:** Base64 encoded PNG image (string)

## 🔧 Environment Variables

The API doesn't require any special environment variables, but you can set:

- `PORT`: Port number (default: 8000)
- `LOG_LEVEL`: Logging level (default: INFO)

## 🚨 Troubleshooting

### Common Issues

1. **Timeout Errors:**
   - Ensure your deployment platform allows requests longer than 30 seconds
   - Railway and Render have good timeout support

2. **Memory Issues:**
   - The API uses pandas and matplotlib which can be memory-intensive
   - Consider upgrading to a plan with more RAM

3. **Docker Build Failures:**
   - Ensure you have enough disk space
   - Try building with `--no-cache` flag

4. **Import Errors:**
   - Make sure all dependencies are in `requirements.txt`
   - Some platforms require specific dependency versions

### Getting Help

1. Check the logs in your deployment platform
2. Test locally first with `python run.py`
3. Use the test script: `python test_api.py`

## 📊 Performance Tips

1. **Caching:** Consider adding Redis for caching scraped data
2. **CDN:** Use a CDN for serving static assets
3. **Database:** For production, consider storing scraped data in a database
4. **Rate Limiting:** Add rate limiting for production use

## 🔒 Security Considerations

1. **CORS:** The API allows all origins by default. Restrict in production
2. **File Uploads:** Validate file types and sizes
3. **Rate Limiting:** Implement rate limiting for production
4. **Authentication:** Add authentication if needed

## 📈 Monitoring

Consider adding monitoring for:
- Response times
- Error rates
- Memory usage
- Request volume

Popular options:
- Sentry (error tracking)
- DataDog (monitoring)
- New Relic (APM) 