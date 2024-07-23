package main

import (
	"bytes"
	"context"
	"encoding/binary"
	"flag"
	"fmt"
	"log"
	"sync"
	"time"

	triton "github.com/triton-inference-server/client/src/grpc_generated/go/grpc-client"

	"google.golang.org/grpc"
)

const (
	inputSize  = 1
	outputSize = 1
)

type Flags struct {
	ModelName    string
	ModelVersion string
	URL          string
}

func parseFlags() Flags {
	var flags Flags
	flag.StringVar(&flags.ModelName, "m", "global_dnn", "Name of model being served. (Required)")
	flag.StringVar(&flags.ModelVersion, "x", "", "Version of model. Default: Latest Version.")
	flag.StringVar(&flags.URL, "u", "10.117.3.165:8001", "Inference Server URL. Default: 10.117.3.165:8001")
	flag.Parse()
	return flags
}

func ModelInferRequest(client triton.GRPCInferenceServiceClient, modelName string, modelVersion string) *triton.ModelInferResponse {
	ctx, cancel := context.WithTimeout(context.Background(), 4*time.Millisecond)
	defer cancel()

	inputData := map[string]interface{}{
		"A": []float32{1.0},
		//		"B": []string{"string1"},
		"C": []float32{2.0},
		"D": []float32{3.0},
		"E": []float32{4.0},
		//		"F": []string{"string2"},
		"G": []float32{5.0},
		"H": []float32{6.0},
		//		"I": []string{"string3"},
		"J": []float32{7.0},
		"K": []float32{8.0},
		//		"L": []string{"string4"},
		"M": []float32{9.0},
		"N": []float32{10.0},
		//		"O": []string{"string5"},
		"P": []float32{11.0},
		"Q": []float32{12.0},
		//		"R": []string{"string6"},
		//		"S": []string{"string7"},
		"T": []float32{13.0},
		"U": []float32{14.0},
		"V": []float32{15.0},
		"W": []float32{16.0},
	}

	var inferInputs []*triton.ModelInferRequest_InferInputTensor
	for name, value := range inputData {
		dtype, shape := getDataTypeAndShape(value)
		inferInput := &triton.ModelInferRequest_InferInputTensor{
			Name:     name,
			Datatype: dtype,
			Shape:    shape,
		}
		inferInputs = append(inferInputs, inferInput)
	}

	inferOutputs := []*triton.ModelInferRequest_InferRequestedOutputTensor{
		&triton.ModelInferRequest_InferRequestedOutputTensor{
			Name: "output_0",
		},
	}

	modelInferRequest := triton.ModelInferRequest{
		ModelName:    modelName,
		ModelVersion: modelVersion,
		Inputs:       inferInputs,
		Outputs:      inferOutputs,
	}

	for _, value := range inputData {
		rawInput := preprocess(value)
		modelInferRequest.RawInputContents = append(modelInferRequest.RawInputContents, rawInput)
	}

	// fmt.Println("inferInputs: ", inferInputs)

	modelInferResponse, err := client.ModelInfer(ctx, &modelInferRequest)
	if err != nil {
		log.Fatalf("Error processing InferRequest: %v", err)
	}
	return modelInferResponse
}

func getDataTypeAndShape(data interface{}) (string, []int64) {
	switch v := data.(type) {
	case []float32:
		return "FP32", []int64{1, int64(len(v))}
	default:
		log.Fatalf("Unsupported data type: %T", v)
	}
	return "", nil
}

func preprocess(data interface{}) []byte {
	var buf bytes.Buffer
	switch v := data.(type) {
	case []float32:
		for _, val := range v {
			binary.Write(&buf, binary.LittleEndian, val)
		}
	case []string:
		for _, val := range v {
			buf.WriteString(val)
		}
	default:
		log.Fatalf("Unsupported data type: %T", v)
	}
	return buf.Bytes()
}

func main() {
	FLAGS := parseFlags()
	fmt.Println("FLAGS:", FLAGS)

	conn, err := grpc.Dial(FLAGS.URL, grpc.WithInsecure())
	if err != nil {
		log.Fatalf("Couldn't connect to endpoint %s: %v", FLAGS.URL, err)
	}
	defer conn.Close()

	client := triton.NewGRPCInferenceServiceClient(conn)
	var wg sync.WaitGroup
	numRequests := 10
	wg.Add(numRequests)

	for i := 0; i < numRequests; i++ {
		go func() {
			defer wg.Done()
			inferResponse := ModelInferRequest(client, FLAGS.ModelName, FLAGS.ModelVersion)

			outputBytes := inferResponse.RawOutputContents[0]
			outputData := make([]float32, len(outputBytes)/4)
			for i := range outputData {
				outputData[i] = readFloat32(outputBytes[i*4 : (i+1)*4])
			}
			fmt.Println("Output:", outputData)
		}()
	}

	wg.Wait()
}

func readFloat32(fourBytes []byte) float32 {
	var val float32
	binary.Read(bytes.NewReader(fourBytes), binary.LittleEndian, &val)
	return val
}
