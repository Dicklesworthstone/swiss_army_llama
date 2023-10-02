from decouple import config
SWISS_ARMY_LLAMA_SERVER_LISTEN_PORT = config("SWISS_ARMY_LLAMA_SERVER_LISTEN_PORT", default=8089, cast=int)
UVICORN_NUMBER_OF_WORKERS = config("UVICORN_NUMBER_OF_WORKERS", default=3, cast=int)

option = {
    "host": "0.0.0.0",
    "port": SWISS_ARMY_LLAMA_SERVER_LISTEN_PORT,
    "workers": UVICORN_NUMBER_OF_WORKERS
}
