package tensor

import (
	"math"
	"testing"
)

func TestF16toF32(t *testing.T) {
	cases := []struct {
		h    uint16
		want float32
	}{
		{0x0000, 0.0},
		{0x3C00, 1.0},
		{0xBC00, -1.0},
		{0x4000, 2.0},
		{0x4200, 3.0},
		{0x3800, 0.5},
	}
	for _, c := range cases {
		got := F16toF32(c.h)
		if math.Abs(float64(got-c.want)) > 1e-6 {
			t.Errorf("F16toF32(0x%04x) = %v, want %v", c.h, got, c.want)
		}
	}
}

func TestMatMul(t *testing.T) {
	// 2x3 @ 3x2 = 2x2
	a := []float32{1, 2, 3, 4, 5, 6}
	b := []float32{1, 2, 3, 4, 5, 6}
	c := make([]float32, 4)
	MatMul(a, b, c, 2, 3, 2)
	want := []float32{22, 28, 49, 64}
	for i, v := range c {
		if v != want[i] {
			t.Errorf("c[%d] = %v, want %v", i, v, want[i])
		}
	}
}

func TestZCRMSNorm(t *testing.T) {
	x := []float32{1, 2, 3, 4}
	scale := []float32{0, 0, 0, 0} // scale init = 0, so output = x / RMS(x)
	out := make([]float32, 4)
	ZCRMSNorm(x, scale, out, 1e-6)
	// RMS = sqrt(mean(1,4,9,16)) = sqrt(7.5) ≈ 2.7386
	want := float32(math.Sqrt(7.5))
	for i, v := range out {
		expected := x[i] / want
		if math.Abs(float64(v-expected)) > 1e-5 {
			t.Errorf("out[%d] = %v, want %v", i, v, expected)
		}
	}
}

func TestSoftmaxRow(t *testing.T) {
	row := []float32{1, 2, 3}
	SoftmaxRow(row)
	var sum float32
	for _, v := range row {
		sum += v
	}
	if math.Abs(float64(sum-1)) > 1e-6 {
		t.Errorf("softmax does not sum to 1: got %v", sum)
	}
	// Verify monotonic.
	if !(row[0] < row[1] && row[1] < row[2]) {
		t.Errorf("softmax not monotonic: %v", row)
	}
}

func TestArgMax(t *testing.T) {
	if got := ArgMax([]float32{1, 5, 3, 2}); got != 1 {
		t.Errorf("got %d, want 1", got)
	}
}
