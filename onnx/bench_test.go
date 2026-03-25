package onnx

import (
	"os"
	"path/filepath"
	"testing"
)

func benchModel(b *testing.B, name string) {
	modelPath := filepath.Join(testdataDir, "models", name, "model.onnx")
	dataDir := filepath.Join(testdataDir, "models", name, "test_data_set_0")
	modelBytes, err := os.ReadFile(modelPath)
	if err != nil {
		b.Skip(err)
	}
	sess, err := NewSession(modelBytes)
	if err != nil {
		b.Fatal(err)
	}
	inputs, err := loadInputs(dataDir)
	if err != nil {
		b.Fatal(err)
	}
	// warmup
	sess.RunWithNames(inputs)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sess.RunWithNames(inputs)
	}
}

func BenchmarkMNIST(b *testing.B)            { benchModel(b, "mnist") }
func BenchmarkSqueezeNet(b *testing.B)       { benchModel(b, "squeezenet") }
func BenchmarkMobileNetV2(b *testing.B)      { benchModel(b, "mobilenetv2") }
func BenchmarkShuffleNet(b *testing.B)       { benchModel(b, "shufflenet") }
func BenchmarkResNet18(b *testing.B)         { benchModel(b, "resnet18") }
func BenchmarkGoogLeNet(b *testing.B)        { benchModel(b, "googlenet") }
func BenchmarkDenseNet121(b *testing.B)      { benchModel(b, "densenet121") }
func BenchmarkEfficientNetLite4(b *testing.B) { benchModel(b, "efficientnet_lite4") }
func BenchmarkVGG16BN(b *testing.B)           { benchModel(b, "vgg16bn") }
func BenchmarkTinyYOLOv2(b *testing.B)        { benchModel(b, "tinyyolov2") }
func BenchmarkSuperRes(b *testing.B)          { benchModel(b, "superres") }
func BenchmarkEmotion(b *testing.B)           { benchModel(b, "emotion") }
func BenchmarkCandy(b *testing.B)             { benchModel(b, "candy") }
func BenchmarkMosaic(b *testing.B)            { benchModel(b, "mosaic") }
func BenchmarkUltraface(b *testing.B)         { benchModel(b, "ultraface") }
func BenchmarkAgeGoogleNet(b *testing.B)      { benchModel(b, "age_googlenet") }
func BenchmarkInceptionV2(b *testing.B)       { benchModel(b, "inception_v2") }
func BenchmarkSileroVAD(b *testing.B)         { benchModel(b, "silero_vad") }
func BenchmarkDEIMv2Atto(b *testing.B)        { benchModel(b, "deimv2_atto") }
func BenchmarkDEIMv2N(b *testing.B)           { benchModel(b, "deimv2_n") }
func BenchmarkDEIMv2S(b *testing.B)           { benchModel(b, "deimv2_s") }
