module github.com/open-ai-org/tesseract

go 1.25.0

require (
	github.com/open-ai-org/gguf v0.0.0
	github.com/open-ai-org/helix v0.0.0
	github.com/open-ai-org/mongoose v0.0.0
	github.com/open-ai-org/needle v0.0.0
	github.com/open-ai-org/tokenizer v0.0.0
)

replace (
	github.com/open-ai-org/gguf => ../gguf
	github.com/open-ai-org/helix => ../helix
	github.com/open-ai-org/mongoose => ../mongoose
	github.com/open-ai-org/needle => ../needle
	github.com/open-ai-org/tokenizer => ../tokenizer
)
