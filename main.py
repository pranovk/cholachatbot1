from fastapi import WebSocket, APIRouter, Request, FastAPI
import json
from cholaOpenAIChat import prompt, endflow, query_engine
from uvicorn import run
from fastapi.middleware.cors import CORSMiddleware
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from search_suggestion import SearchSuggestion

ss = SearchSuggestion()
app = FastAPI()
data = {
    "suggestionList": []
    }

origins = ["*"]

# middleware for cors and headers

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    )


@app.websocket("/search")
async def search(WebSocket: WebSocket):
    await WebSocket.accept()
    while True:
        query = await WebSocket.receive_text()
        data.get("suggestionList").append(str(query).lower())
        ss.insert(str(query).lower())
        if (prompt != endflow):
            response = query_engine.query(query)
            await WebSocket.send_text(str(response))
        else:
            await WebSocket.send_text("Exiting")
            break


@app.websocket("/")
async def main(WebSocket: WebSocket):
    await WebSocket.accept()
    while True:
        query = await WebSocket.receive_text()
        response = ss.search(str(query).lower(), max_suggestions=20)
        await WebSocket.send_json(response)


@app.post("/suggestions")
async def policyQuery(WebSocket: Request):
    query = await WebSocket.body()
    query = json.loads(query.decode("UTF-8"))["query"]
    response = ss.search(str(query).lower(), max_suggestions=20)
    return JSONResponse(status_code=200, content=jsonable_encoder({ "result": response }))


@app.post("/policyQuery")
async def policyQuery(WebSocket: Request):
    query = await WebSocket.body()
    query = json.loads(query.decode("UTF-8"))["query"]
    data.get("suggestionList").append(str(query).lower())
    ss.insert(str(query).lower())
    if (prompt != endflow):
        print(query)
        response = query_engine.query(query)
        return JSONResponse(status_code=200, content=jsonable_encoder({ "result": response.response }))
    else:
        return JSONResponse(status_code=200, content=jsonable_encoder({ "result": "Exiting" }))


@app.on_event("startup")
def startup_event():
    file = open('./Chola_policy_index/searchSuggestion.json', mode="rb")
    bulk_data = json.loads(file.read().decode())
    data["suggestionList"] = bulk_data["suggestionList"]
    ss.batch_insert(bulk_data["suggestionList"])


@app.on_event("shutdown")
def shutdowm_event():
    with open('./Chola_policy_index/searchSuggestion.json', 'w') as f:
        f.write(json.dumps(data))


if __name__ == "__main__":
    run("main:app", host="0.0.0.0", reload=True)
