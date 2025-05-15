from voice2text import celery_app

if __name__ == "__main__":
    celery_app.worker_main(["worker", "--loglevel=info", "-Q", "transcription,summarization"]) 