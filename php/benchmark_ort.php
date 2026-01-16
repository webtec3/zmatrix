<?php
use ZMatrix\ZTensor;
use ORT\Tensor\Transient;

define('ITERATIONS', 50);
define('MATRIX_SIZE', 100);
define('VECTOR_SIZE', 3000);

const GREEN = "\033[92m";
const BLUE = "\033[94m";
const YELLOW = "\033[93m";
const CYAN = "\033[96m";
const RESET = "\033[0m";
const BOLD = "\033[1m";

class Benchmark {
    private $results = [];
    
    private function fmt($ms) {
        return $ms < 1 ? sprintf("%.3f µs", $ms * 1000) : ($ms < 1000 ? sprintf("%.3f ms", $ms) : sprintf("%.3f s", $ms / 1000));
    }
    
    private function make2dArray($rows, $cols, $val = 0.5) {
        $arr = [];
        for ($i = 0; $i < $rows; $i++) {
            $arr[] = array_fill(0, $cols, $val);
        }
        return $arr;
    }
    
    public function test1_Addition() {
        echo BOLD . CYAN . "\n[TEST 1] Vector Addition (" . VECTOR_SIZE . " elements)" . RESET . "\n";
        echo str_repeat("-", 80) . "\n";
        
        // ZMatrix
        echo "ZMatrix: ";
        gc_collect_cycles();
        $start = microtime(true);
        for ($i = 0; $i < ITERATIONS; $i++) {
            $v1 = new ZTensor([VECTOR_SIZE]);
            $v2 = new ZTensor([VECTOR_SIZE]);
            $r = $v1->add($v2);
        }
        $t_z = (microtime(true) - $start) * 1000;
        echo GREEN . $this->fmt($t_z / ITERATIONS) . RESET . "\n";
        $this->results['add']['z'] = $t_z / ITERATIONS;
        
        // ORT
        echo "ORT:     ";
        gc_collect_cycles();
        $start = microtime(true);
        for ($i = 0; $i < ITERATIONS; $i++) {
            $data = array_fill(0, VECTOR_SIZE, 1.0);
            $v1 = new Transient([VECTOR_SIZE], $data, ORT\Tensor::FLOAT32);
            $v2 = new Transient([VECTOR_SIZE], $data, ORT\Tensor::FLOAT32);
            $r = \ORT\Math\add($v1, $v2);
        }
        $t_o = (microtime(true) - $start) * 1000;
        echo BLUE . $this->fmt($t_o / ITERATIONS) . RESET . "\n";
        $this->results['add']['o'] = $t_o / ITERATIONS;
        
        $ratio = $this->results['add']['z'] / $this->results['add']['o'];
        echo "Winner: " . ($ratio < 1 ? GREEN . "ZMatrix" . RESET : BLUE . "ORT" . RESET) . " (" . sprintf("%.2fx", max($ratio, 1/$ratio)) . ")\n";
    }
    
    public function test2_Multiplication() {
        echo BOLD . CYAN . "\n[TEST 2] Matrix Multiply (dot product, " . MATRIX_SIZE . "x" . MATRIX_SIZE . ")" . RESET . "\n";
        echo str_repeat("-", 80) . "\n";
        
        // ZMatrix
        echo "ZMatrix: ";
        gc_collect_cycles();
        $start = microtime(true);
        for ($i = 0; $i < ITERATIONS; $i++) {
            $m1 = new ZTensor([MATRIX_SIZE, MATRIX_SIZE]);
            $m2 = new ZTensor([MATRIX_SIZE, MATRIX_SIZE]);
            $r = $m1->dot($m2);
        }
        $t_z = (microtime(true) - $start) * 1000;
        echo GREEN . $this->fmt($t_z / ITERATIONS) . RESET . "\n";
        $this->results['mul']['z'] = $t_z / ITERATIONS;
        
        // ORT
        echo "ORT:     ";
        gc_collect_cycles();
        $start = microtime(true);
        for ($i = 0; $i < ITERATIONS; $i++) {
            $data = $this->make2dArray(MATRIX_SIZE, MATRIX_SIZE, 0.5);
            $m1 = new Transient([MATRIX_SIZE, MATRIX_SIZE], $data, ORT\Tensor::FLOAT32);
            $m2 = new Transient([MATRIX_SIZE, MATRIX_SIZE], $data, ORT\Tensor::FLOAT32);
            $r = \ORT\Math\matmul($m1, $m2);
        }
        $t_o = (microtime(true) - $start) * 1000;
        echo BLUE . $this->fmt($t_o / ITERATIONS) . RESET . "\n";
        $this->results['mul']['o'] = $t_o / ITERATIONS;
        
        $ratio = $this->results['mul']['z'] / $this->results['mul']['o'];
        echo "Winner: " . ($ratio < 1 ? GREEN . "ZMatrix" . RESET : BLUE . "ORT" . RESET) . " (" . sprintf("%.2fx", max($ratio, 1/$ratio)) . ")\n";
    }
    
    public function test3_ElementWiseMultiply() {
        echo BOLD . CYAN . "\n[TEST 3] Element-wise Multiply (scalar)" . RESET . "\n";
        echo str_repeat("-", 80) . "\n";
        
        // ZMatrix
        echo "ZMatrix: ";
        gc_collect_cycles();
        $start = microtime(true);
        for ($i = 0; $i < 50; $i++) {
            $v = new ZTensor([VECTOR_SIZE]);
            $r = $v->scalarMultiply(2.5);
        }
        $t_z = (microtime(true) - $start) * 1000;
        echo GREEN . $this->fmt($t_z / 50) . RESET . "\n";
        $this->results['elem']['z'] = $t_z / 50;
        
        // ORT
        echo "ORT:     ";
        gc_collect_cycles();
        $start = microtime(true);
        for ($i = 0; $i < 50; $i++) {
            $data = array_fill(0, VECTOR_SIZE, 0.5);
            $v = new Transient([VECTOR_SIZE], $data, ORT\Tensor::FLOAT32);
            $r = \ORT\Math\multiply($v, 2.5);
        }
        $t_o = (microtime(true) - $start) * 1000;
        echo BLUE . $this->fmt($t_o / 50) . RESET . "\n";
        $this->results['elem']['o'] = $t_o / 50;
        
        $ratio = $this->results['elem']['z'] / $this->results['elem']['o'];
        echo "Winner: " . ($ratio < 1 ? GREEN . "ZMatrix" . RESET : BLUE . "ORT" . RESET) . " (" . sprintf("%.2fx", max($ratio, 1/$ratio)) . ")\n";
    }
    
    public function test4_MatrixScale() {
        echo BOLD . CYAN . "\n[TEST 4] Matrix Scale (150x150 multiply by 3.5)" . RESET . "\n";
        echo str_repeat("-", 80) . "\n";
        
        // ZMatrix
        echo "ZMatrix: ";
        gc_collect_cycles();
        $start = microtime(true);
        for ($i = 0; $i < 40; $i++) {
            $m = new ZTensor([150, 150]);
            $r = $m->scalarMultiply(3.5);
        }
        $t_z = (microtime(true) - $start) * 1000;
        echo GREEN . $this->fmt($t_z / 40) . RESET . "\n";
        $this->results['scale']['z'] = $t_z / 40;
        
        // ORT
        echo "ORT:     ";
        gc_collect_cycles();
        $start = microtime(true);
        for ($i = 0; $i < 40; $i++) {
            $data = $this->make2dArray(150, 150, 0.5);
            $m = new Transient([150, 150], $data, ORT\Tensor::FLOAT32);
            $r = \ORT\Math\multiply($m, 3.5);
        }
        $t_o = (microtime(true) - $start) * 1000;
        echo BLUE . $this->fmt($t_o / 40) . RESET . "\n";
        $this->results['scale']['o'] = $t_o / 40;
        
        $ratio = $this->results['scale']['z'] / $this->results['scale']['o'];
        echo "Winner: " . ($ratio < 1 ? GREEN . "ZMatrix" . RESET : BLUE . "ORT" . RESET) . " (" . sprintf("%.2fx", max($ratio, 1/$ratio)) . ")\n";
    }
    
    public function test5_ElementWiseAdd() {
        echo BOLD . CYAN . "\n[TEST 5] Element-wise Addition (large tensors)" . RESET . "\n";
        echo str_repeat("-", 80) . "\n";
        
        // ZMatrix
        echo "ZMatrix: ";
        gc_collect_cycles();
        $start = microtime(true);
        for ($i = 0; $i < 30; $i++) {
            $m1 = new ZTensor([250, 250]);
            $m2 = new ZTensor([250, 250]);
            $r = $m1->add($m2);
        }
        $t_z = (microtime(true) - $start) * 1000;
        echo GREEN . $this->fmt($t_z / 30) . RESET . "\n";
        $this->results['ewadd']['z'] = $t_z / 30;
        
        // ORT
        echo "ORT:     ";
        gc_collect_cycles();
        $start = microtime(true);
        for ($i = 0; $i < 30; $i++) {
            $data = $this->make2dArray(250, 250, 0.5);
            $m1 = new Transient([250, 250], $data, ORT\Tensor::FLOAT32);
            $m2 = new Transient([250, 250], $data, ORT\Tensor::FLOAT32);
            $r = \ORT\Math\add($m1, $m2);
        }
        $t_o = (microtime(true) - $start) * 1000;
        echo BLUE . $this->fmt($t_o / 30) . RESET . "\n";
        $this->results['ewadd']['o'] = $t_o / 30;
        
        $ratio = $this->results['ewadd']['z'] / $this->results['ewadd']['o'];
        echo "Winner: " . ($ratio < 1 ? GREEN . "ZMatrix" . RESET : BLUE . "ORT" . RESET) . " (" . sprintf("%.2fx", max($ratio, 1/$ratio)) . ")\n";
    }
    
    public function summary() {
        echo BOLD . CYAN . "\n═══════════════════════════════════════════════════════════════════════════════\n";
        echo "                            FINAL SCORE\n";
        echo "═══════════════════════════════════════════════════════════════════════════════" . RESET . "\n\n";
        
        $z = 0;
        $o = 0;
        
        foreach ($this->results as $test => $libs) {
            if (isset($libs['z']) && isset($libs['o'])) {
                if ($libs['z'] < $libs['o']) {
                    $ratio = $libs['o'] / $libs['z'];
                    echo "  ✓ " . str_pad(strtoupper($test), 6) . ": " . GREEN . "ZMatrix" . RESET . " is " . sprintf("%.2fx faster", $ratio) . "\n";
                    $z++;
                } else {
                    $ratio = $libs['z'] / $libs['o'];
                    echo "  ✓ " . str_pad(strtoupper($test), 6) . ": " . BLUE . "ORT" . RESET . " is " . sprintf("%.2fx faster", $ratio) . "\n";
                    $o++;
                }
            }
        }
        
        echo "\n  " . BOLD . "Score:" . RESET . " ZMatrix: " . GREEN . $z . RESET . " | ORT: " . BLUE . $o . RESET . "\n";
        
        if ($z > $o) {
            echo "\n  " . GREEN . BOLD . "🏆 OVERALL WINNER: ZMatrix (Faster for simple operations)" . RESET . "\n";
        } elseif ($o > $z) {
            echo "\n  " . BLUE . BOLD . "🏆 OVERALL WINNER: ORT (Better for complex operations)" . RESET . "\n";
        } else {
            echo "\n  " . YELLOW . BOLD . "⚖️  PERFECT DRAW!" . RESET . "\n";
        }
        
        echo "\n" . str_repeat("═", 80) . "\n";
    }
    
    public function run() {
        echo BOLD . "\n╔════════════════════════════════════════════════════════════════════════════════╗\n";
        echo "║                  HEAVY BENCHMARK: ORT vs ZMatrix                           ║\n";
        echo "╚════════════════════════════════════════════════════════════════════════════╝" . RESET . "\n\n";
        echo "PHP " . phpversion() . " | ORT: " . (extension_loaded('ort') ? GREEN . "✓" . RESET : "✗") . " | ZMatrix: " . (extension_loaded('zmatrix') ? GREEN . "✓" . RESET : "✗") . "\n";
        
        $this->test1_Addition();
        $this->test2_Multiplication();
        $this->test3_ElementWiseMultiply();
        $this->test4_MatrixScale();
        $this->test5_ElementWiseAdd();
        $this->summary();
        
        echo "\n✅ Benchmark finished at " . date('H:i:s') . "\n\n";
    }
}

(new Benchmark())->run();
