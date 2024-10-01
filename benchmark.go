package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"math"
	"os"

	"github.com/ollama/ollama/api"
)

type timing struct {
	promptRate []float64
	evalRate   []float64
	record     bool
}

func (stats *timing) reply(resp api.GenerateResponse) error {
	if resp.Done {
		if stats.record {
			if resp.Metrics.PromptEvalDuration > 0 {
				rate := float64(resp.Metrics.PromptEvalCount) / resp.Metrics.PromptEvalDuration.Seconds()
				//fmt.Printf("prompt eval duration: %s\n", resp.Metrics.PromptEvalDuration)
				//fmt.Printf("prompt eval rate:     %.2f tokens/s\n", rate)

				stats.promptRate = append(stats.promptRate, rate)
			}

			if resp.Metrics.EvalDuration > 0 {
				rate := float64(resp.Metrics.EvalCount) / resp.Metrics.EvalDuration.Seconds()
				stats.evalRate = append(stats.evalRate, rate)
			}
		} else {
			stats.record = true
		}
	}

	return nil
}

type Prompt struct {
	P []string `json:"prompt"`
}

func main() {
	model := flag.String("model", "llama3.1", "Model to benchmark")
	benchPrompt := flag.Bool("prompt", false, "Benchmark a long prompt (vs. long generation)")
	runs := flag.Int("runs", 10, "Number of runs")

	flag.Parse()

	ctx := context.Background()
	client, err := api.ClientFromEnvironment()
	if err != nil {
		panic(err)
	}

	promptFile := "generate"
	if *benchPrompt {
		promptFile = "prompt"
	}

	f, err := os.Open(promptFile)
	if err != nil {
		panic(err)
	}
	defer f.Close()

	var prompt Prompt
	if err := json.NewDecoder(f).Decode(&prompt); err != nil {
		panic(err)
	}

	stream := false
	req := api.GenerateRequest{
		Model:  *model,
		Stream: &stream,
		Options: map[string]any{
			"temperature": 0,
			"seed":        0,
		}}

	stats := timing{}

	for i := range *runs + 1 {
		req.Prompt = prompt.P[i%len(prompt.P)]
		err = client.Generate(ctx, &req, stats.reply)
		if err != nil {
			panic(err)
		}
	}

	if *benchPrompt {
		fmt.Print("prompt ")
		printStats(stats.promptRate)
	} else {
		fmt.Print("eval ")
		printStats(stats.evalRate)
	}
}

func printStats(rates []float64) {
	var minRate float64 = math.MaxFloat64
	var maxRate float64
	var sum float64

	for _, rate := range rates {
		sum += rate
		if minRate > rate {
			minRate = rate
		}
		if maxRate < rate {
			maxRate = rate
		}
	}
	fmt.Printf("average: %.2f min: %.2f max: %.2f\n\n", sum/float64(len(rates)), minRate, maxRate)

	for _, rate := range rates {
		fmt.Printf("%.2f\n", rate)
	}
}
