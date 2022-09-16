build:
	docker build -t sarcopeniaai -f ./Dockerfile .
	
server: build
	docker run --rm -it -p 5000:5000 sarcopeniaai python -m sarcopenia_ai.apps.server.run_local_server