from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, ValidationError, validator
from Database import Database

app = FastAPI()

templates = Jinja2Templates(directory='templates')


def results_to_json(name, probabilities, labels=None, img=""):
    if labels is None:
        labels = [""]
    return {
        "name": name,
        "probabilities": probabilities,
        "image": img,
        "labels": labels
    }


@app.get("/")
def home(request: Request):
    """
    Return html template render for home page form
    """

    return templates.TemplateResponse('index.html', {"request": request})


class DataRequest(BaseModel):
    """
    Class used for pypandic validation
    """
    word: str
    host: str = "localhost"
    user: str = "roots"
    port: int = 3306
    pass_word: str = "BAPInternAI2021@"
    database: str = "Test"

    # @validator('word')
    # def validate_word(cls, word):
    #     assert word == "", f'Invalid word. Word cannot be blank!'
    #     return word


@app.post("/getdata")
async def get_data_from_api(request: Request,
                            word: str,
                            host: str = "localhost",
                            user: str = "roots",
                            pass_word: str = "BAPInternAI2021@",
                            port: int = 3306,
                            database: str = "Test"
                            ):
    """
    Requires an image file upload and Optional image size parameter.
    Intended for API users.
    Return: 
    """

    try:
        data_request = DataRequest(word=word)
    except ValidationError as e:
        return JSONResponse(content=e.errors(), status_code=422)
    con = Database(host=host, user=user, password=pass_word, port=port, database=database)
    sql = 'SELECT * FROM SYNONYMS WHERE WORD="' + word + '"'
    data = con.select(sql)
    for item in data:
        print(item)
    return {word: data}


if __name__ == '__main__':
    import uvicorn

    app_str = 'server:app'
    uvicorn.run(app_str, host='localhost', port=8000, reload=True, workers=1)
