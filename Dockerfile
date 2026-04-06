FROM golang:1.22-alpine AS builder

WORKDIR /app

COPY go.mod go.sum ./
RUN go mod download

COPY . .
RUN CGO_ENABLED=0 GOOS=linux go build -trimpath -o llm-proxy ./cmd/server

FROM alpine:3.19

RUN apk --no-cache add ca-certificates tzdata

WORKDIR /app
COPY --from=builder /app/llm-proxy .

EXPOSE 8080

ENTRYPOINT ["./llm-proxy"]
CMD ["--config", "/app/config/config.yaml"]
