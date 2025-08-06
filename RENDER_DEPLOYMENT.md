# 🚀 Deploy to Render

## Prerequisites
- GitHub account connected to Render
- Your repository pushed to GitHub

## Step-by-Step Deployment

### 1. **Push to GitHub**
```bash
git add .
git commit -m "Prepare for Render deployment"
git push origin main
```

### 2. **Deploy on Render**

1. **Go to [Render Dashboard](https://dashboard.render.com/)**
2. **Click "New +" → "Web Service"**
3. **Connect your GitHub repository**
4. **Configure the service:**

   - **Name:** `data-analyst-agent-api`
   - **Environment:** `Python 3`
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `uvicorn api.main:app --host 0.0.0.0 --port $PORT`
   - **Plan:** Free (or paid for better performance)

5. **Click "Create Web Service"**

### 3. **Environment Variables (Optional)**
If you have any API keys, add them in Render dashboard:
- Go to your service → Environment
- Add any required environment variables

### 4. **Wait for Deployment**
- Render will automatically build and deploy your app
- You'll get a URL like: `https://your-app-name.onrender.com`

### 5. **Test Your API**
- **Swagger UI:** `https://your-app-name.onrender.com/docs`
- **Health Check:** `https://your-app-name.onrender.com/`
- **API Endpoint:** `https://your-app-name.onrender.com/api/`

## API Endpoints

### File Upload Endpoint
```bash
curl -X POST "https://your-app-name.onrender.com/api/" \
  -F "questions_txt=@question.txt"
```

### JSON Endpoint
```bash
curl -X POST "https://your-app-name.onrender.com/api/json/" \
  -H "Content-Type: application/json" \
  -d '{"question": "Your question here"}'
```

## Troubleshooting

### Common Issues:
1. **Build fails:** Check the build logs in Render dashboard
2. **Import errors:** Make sure all dependencies are in `requirements.txt`
3. **Port issues:** The `$PORT` environment variable is automatically set by Render
4. **Timeout:** Render has a 30-second timeout for free tier

### Performance Tips:
- **Free tier:** Limited to 750 hours/month
- **Paid tier:** Better performance and no timeout limits
- **Auto-deploy:** Enabled by default - deploys on every push to main branch

## Monitoring
- Check deployment logs in Render dashboard
- Monitor performance and errors
- Set up alerts if needed

## Update Your API URL
Once deployed, update your evaluation configuration:
```yaml
providers:
  - id: https
    config:
      url: https://your-app-name.onrender.com/api/  # Your Render URL
      method: POST
      body: file://question.txt
      transformResponse: json
``` 