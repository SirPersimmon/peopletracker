FROM gcr.io/kaggle-gpu-images/python AS builder
WORKDIR /app
RUN --mount=target=/data,z /data/build-openpose.sh cpu

FROM gcr.io/kaggle-gpu-images/python

WORKDIR /app

ENV OPENPOSE_PATH="/app/openpose"

COPY --from=builder /app/openpose/build ./openpose/build
COPY --from=builder /app/openpose/models ./openpose/models

RUN apt update && apt install libgoogle-glog0v5 && \
    uv pip install --system pykalman streamlit && \
    git clone https://github.com/SirPersimmon/peopletracker.git /tmp/peopletracker && \
    mv /tmp/peopletracker/src/peopletracker . && \
    rm -rf /tmp/peopletracker

EXPOSE 8501

ENTRYPOINT ["streamlit", "run"]

CMD ["/app/peopletracker/webui.py"]
