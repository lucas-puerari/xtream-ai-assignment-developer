import uvicorn
from fastapi import FastAPI

from interfaces.routers import router as diamond_router


app = FastAPI()
app.include_router(diamond_router)


@app.get("/")
def hello_world():
    return {"Message": "Hello Diamonds"}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
