# Data Analyst Agent API

A FastAPI-based data analyst agent that can scrape, analyze, and visualize data from web sources using LLMs.

## Features

- Web scraping from Wikipedia and other sources
- Data analysis and statistical calculations
- Data visualization with matplotlib/seaborn
- RESTful API with file upload support
- Docker containerization support

## API Endpoint

**POST** `/api/`

Accepts a file upload containing a data analysis question and returns a JSON array with exactly 4 elements:
1. Count (integer) - Number of items matching criteria
2. Movie name (string) - Name of the earliest movie matching criteria  
3. Correlation (float) - Correlation coefficient between specified columns
4. Plot image (string) - Base64 encoded PNG image

### Example Request

```bash
curl "https://your-api-domain.com/api/" -F "@question.txt"
```

Where `question.txt` contains:
```
Scrape the list of highest grossing films from Wikipedia. It is at the URL:
https://en.wikipedia.org/wiki/List_of_highest-grossing_films

Answer the following questions and respond with a JSON array of strings containing the answer.

1. How many $2 bn movies were released before 2000?
2. Which is the earliest film that grossed over $1.5 bn?
3. What's the correlation between the Rank and Peak?
4. Draw a scatterplot of Rank and Peak along with a dotted red regression line through it.
   Return as a base-64 encoded data URI, `"data:image/png;base64,iVBORw0KG..."` under 100,000 bytes.
```

### Example Response

```json
[1, "Titanic", 0.485782, "data:image/png;base64,iVBORw0KG..."]
```

## Deployment

### Local Development

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the application:
```bash
python run.py
```

The API will be available at `http://localhost:8000`

### Docker Deployment

1. Build the Docker image:
```bash
docker build -t data-analyst-api .
```

2. Run the container:
```bash
docker run -p 8000:8000 data-analyst-api
```

### Cloud Deployment

#### Railway
1. Connect your GitHub repository to Railway
2. Railway will automatically detect the Dockerfile and deploy
3. Your API will be available at the provided Railway URL

#### Heroku
1. Create a `Procfile`:
```
web: uvicorn api.main:app --host=0.0.0.0 --port=$PORT
```

2. Deploy using Heroku CLI:
```bash
heroku create your-app-name
git push heroku main
```

#### Vercel
1. Create a `vercel.json`:
```json
{
  "version": 2,
  "builds": [
    {
      "src": "run.py",
      "use": "@vercel/python"
    }
  ],
  "routes": [
    {
      "src": "/(.*)",
      "dest": "run.py"
    }
  ]
}
```

2. Deploy using Vercel CLI:
```bash
vercel --prod
```

## Testing

Run the test script to verify the API functionality:

```bash
python test_api.py
```

## Project Structure

```
├── api/
│   ├── main.py          # FastAPI application
│   ├── handler.py       # Request handling logic
│   ├── analyzer.py      # Data analysis functions
│   ├── scraper.py       # Web scraping functionality
│   ├── plotter.py       # Data visualization
│   └── ...
├── requirements.txt     # Python dependencies
├── Dockerfile          # Docker configuration
├── run.py             # Local development runner
└── test_api.py        # API testing script
```

## Dependencies

- FastAPI - Web framework
- Uvicorn - ASGI server
- Pandas - Data manipulation
- Matplotlib/Seaborn - Data visualization
- BeautifulSoup - Web scraping
- Requests - HTTP client
- And more (see requirements.txt)

## License

MIT License 