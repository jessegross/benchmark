package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"math"
	"os"
	"sync"
	"time"

	"github.com/ollama/ollama/api"
	"golang.org/x/sync/semaphore"
)

type timing struct {
	mu           sync.Mutex
	promptTokens int
	promptRate   []float64
	evalTokens   int
	evalRate     []float64
	record       bool
}

func (stats *timing) reply(resp api.GenerateResponse) error {
	if resp.Done {
		stats.mu.Lock()
		defer stats.mu.Unlock()

		if stats.record {
			if resp.Metrics.PromptEvalDuration > 0 {
				stats.promptTokens += resp.Metrics.PromptEvalCount
				rate := float64(resp.Metrics.PromptEvalCount) / resp.Metrics.PromptEvalDuration.Seconds()
				stats.promptRate = append(stats.promptRate, rate)
			}

			if resp.Metrics.EvalDuration > 0 {
				stats.evalTokens += resp.Metrics.EvalCount
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
	parallel := flag.Int("parallel", 1, "Runs to do in parallel")

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
			"num_predict": 400,
			"num_ctx": 512,
		}}

	stats := timing{}

	var wg sync.WaitGroup
	sem := semaphore.NewWeighted(int64(*parallel))

	wg.Add(*runs)

	req.Prompt = prompt.P[0]
	err = client.Generate(ctx, &req, stats.reply)
	if err != nil {
		panic(err)
	}

	startTime := time.Now()
	for i := range *runs {
		go func() {
			send := req
			send.Prompt = prompt.P[(i+1)%len(prompt.P)]
			sem.Acquire(ctx, 1)
			err = client.Generate(ctx, &send, stats.reply)
			sem.Release(1)
			if err != nil {
				panic(err)
			}
			wg.Done()
		}()
	}

	wg.Wait()
	totalTime := time.Since(startTime)

	if *benchPrompt {
		fmt.Print("prompt ")
		printStats(stats.promptRate)
		fmt.Printf("\nTPS: %v\n", float32(stats.promptTokens)/float32(totalTime.Seconds()))
	} else {
		fmt.Print("eval ")
		printStats(stats.evalRate)
		fmt.Printf("\nTPS: %v\n", float32(stats.evalTokens)/float32(totalTime.Seconds()))
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
