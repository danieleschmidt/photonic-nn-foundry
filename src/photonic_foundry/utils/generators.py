"""
Code generation utilities for Verilog and testbenches.
"""

from typing import Dict, List, Any, Optional
import logging
from datetime import datetime
import re

logger = logging.getLogger(__name__)


class VerilogGenerator:
    """Advanced Verilog code generation for photonic circuits."""
    
    def __init__(self, target_pdk: str = "skywater130"):
        self.target_pdk = target_pdk
        self.indent_level = 0
        self.generated_modules = set()
        
    def generate_module_header(self, module_name: str, ports: List[Dict[str, str]]) -> str:
        """Generate Verilog module header with ports."""
        header = f"module {module_name} (\n"
        
        port_declarations = []
        for port in ports:
            direction = port.get('direction', 'input')
            width = port.get('width', '1')
            name = port.get('name', 'unknown')
            
            if width == '1':
                port_declarations.append(f"    {direction} {name}")
            else:
                port_declarations.append(f"    {direction} [{width}] {name}")
                
        header += ",\n".join(port_declarations)
        header += "\n);\n\n"
        
        return header
        
    def generate_parameter_section(self, parameters: Dict[str, Any]) -> str:
        """Generate parameter declarations."""
        if not parameters:
            return ""
            
        param_section = "// Parameters\n"
        for name, value in parameters.items():
            if isinstance(value, str):
                param_section += f"parameter {name} = \"{value}\";\n"
            else:
                param_section += f"parameter {name} = {value};\n"
                
        param_section += "\n"
        return param_section
        
    def generate_signal_declarations(self, signals: List[Dict[str, str]]) -> str:
        """Generate internal signal declarations."""
        if not signals:
            return ""
            
        decl_section = "// Internal signals\n"
        for signal in signals:
            signal_type = signal.get('type', 'wire')
            width = signal.get('width', '1')
            name = signal.get('name', 'unknown')
            
            if width == '1':
                decl_section += f"{signal_type} {name};\n"
            else:
                decl_section += f"{signal_type} [{width}] {name};\n"
                
        decl_section += "\n"
        return decl_section
        
    def generate_mzi_array(self, rows: int, cols: int, precision: int = 8) -> str:
        """Generate MZI array instantiation."""
        mzi_code = f"// MZI Array: {rows}x{cols}\n"
        mzi_code += "genvar i, j;\n"
        mzi_code += "generate\n"
        mzi_code += f"    for (i = 0; i < {rows}; i = i + 1) begin: mzi_row\n"
        mzi_code += f"        for (j = 0; j < {cols}; j = j + 1) begin: mzi_col\n"
        mzi_code += "            mzi_unit #(\n"
        mzi_code += f"                .PRECISION({precision}),\n"
        mzi_code += "                .WEIGHT(weights[i][j])\n"
        mzi_code += "            ) mzi_inst (\n"
        mzi_code += "                .clk(clk),\n"
        mzi_code += "                .rst_n(rst_n),\n"
        mzi_code += "                .data_in(layer_input[j]),\n"
        mzi_code += "                .weight_out(mzi_output[i][j])\n"
        mzi_code += "            );\n"
        mzi_code += "        end\n"
        mzi_code += "    end\n"
        mzi_code += "endgenerate\n\n"
        
        return mzi_code
        
    def generate_accumulator_tree(self, inputs: int, precision: int = 8) -> str:
        """Generate efficient accumulator tree."""
        acc_code = f"// Accumulator tree for {inputs} inputs\n"
        
        # Calculate tree depth
        tree_depth = max(1, (inputs - 1).bit_length())
        
        acc_code += f"// Tree depth: {tree_depth}\n"
        
        for level in range(tree_depth):
            level_inputs = inputs >> level
            level_outputs = (level_inputs + 1) >> 1
            
            if level == 0:
                acc_code += f"wire [{precision-1}:0] acc_level_{level} [{level_outputs-1}:0];\n"
            else:
                acc_code += f"wire [{precision-1}:0] acc_level_{level} [{level_outputs-1}:0];\n"
                
            acc_code += "genvar k;\n"
            acc_code += "generate\n"
            acc_code += f"    for (k = 0; k < {level_outputs}; k = k + 1) begin: acc_level_{level}_gen\n"
            
            if level == 0:
                acc_code += "        if (k*2+1 < {}) begin\n".format(inputs)
                acc_code += f"            assign acc_level_{level}[k] = mzi_output[k*2] + mzi_output[k*2+1];\n"
                acc_code += "        end else begin\n"
                acc_code += f"            assign acc_level_{level}[k] = mzi_output[k*2];\n"
                acc_code += "        end\n"
            else:
                prev_level = level - 1
                prev_size = inputs >> prev_level
                acc_code += f"        if (k*2+1 < {prev_size}) begin\n"
                acc_code += f"            assign acc_level_{level}[k] = acc_level_{prev_level}[k*2] + acc_level_{prev_level}[k*2+1];\n"
                acc_code += "        end else begin\n"
                acc_code += f"            assign acc_level_{level}[k] = acc_level_{prev_level}[k*2];\n"
                acc_code += "        end\n"
                
            acc_code += "    end\n"
            acc_code += "endgenerate\n\n"
            
        # Final output assignment
        acc_code += f"assign accumulator_out = acc_level_{tree_depth-1}[0];\n\n"
        
        return acc_code
        
    def generate_pipeline_registers(self, stages: int, data_width: int) -> str:
        """Generate pipeline register stages."""
        pipeline_code = f"// Pipeline registers: {stages} stages\n"
        
        for stage in range(stages):
            pipeline_code += f"reg [{data_width-1}:0] pipeline_stage_{stage};\n"
            pipeline_code += f"reg valid_stage_{stage};\n"
            
        pipeline_code += "\n"
        pipeline_code += "always @(posedge clk or negedge rst_n) begin\n"
        pipeline_code += "    if (!rst_n) begin\n"
        
        for stage in range(stages):
            pipeline_code += f"        pipeline_stage_{stage} <= 0;\n"
            pipeline_code += f"        valid_stage_{stage} <= 1'b0;\n"
            
        pipeline_code += "    end else begin\n"
        pipeline_code += "        // Stage 0: Input\n"
        pipeline_code += "        pipeline_stage_0 <= data_in;\n"
        pipeline_code += "        valid_stage_0 <= valid_in;\n"
        
        for stage in range(1, stages):
            pipeline_code += f"        // Stage {stage}\n"
            pipeline_code += f"        pipeline_stage_{stage} <= pipeline_stage_{stage-1};\n"
            pipeline_code += f"        valid_stage_{stage} <= valid_stage_{stage-1};\n"
            
        pipeline_code += "    end\n"
        pipeline_code += "end\n\n"
        
        # Output assignments
        pipeline_code += f"assign data_out = pipeline_stage_{stages-1};\n"
        pipeline_code += f"assign valid_out = valid_stage_{stages-1};\n\n"
        
        return pipeline_code
        
    def generate_complete_module(self, module_name: str, config: Dict[str, Any]) -> str:
        """Generate complete Verilog module."""
        # Module header
        ports = [
            {'direction': 'input', 'name': 'clk'},
            {'direction': 'input', 'name': 'rst_n'},
            {'direction': 'input', 'width': f"{config.get('input_width', 32)-1}:0", 'name': 'data_in'},
            {'direction': 'input', 'name': 'valid_in'},
            {'direction': 'output', 'width': f"{config.get('output_width', 32)-1}:0", 'name': 'data_out'},
            {'direction': 'output', 'name': 'valid_out'}
        ]\n        \n        verilog_code = self.generate_module_header(module_name, ports)\n        \n        # Parameters\n        parameters = {\n            'INPUT_WIDTH': config.get('input_width', 32),\n            'OUTPUT_WIDTH': config.get('output_width', 32),\n            'PRECISION': config.get('precision', 8),\n            'PIPELINE_STAGES': config.get('pipeline_stages', 3)\n        }\n        verilog_code += self.generate_parameter_section(parameters)\n        \n        # Internal signals\n        signals = [\n            {'type': 'wire', 'width': f"{config.get('precision', 8)-1}:0", 'name': 'mzi_data'},\n            {'type': 'wire', 'width': f"{config.get('output_width', 32)-1}:0", 'name': 'accumulator_out'},\n            {'type': 'reg', 'width': f"{config.get('precision', 8)-1}:0", 'name': 'processed_data'}\n        ]\n        verilog_code += self.generate_signal_declarations(signals)\n        \n        # MZI array\n        if config.get('include_mzi_array', True):\n            rows = config.get('mzi_rows', 4)\n            cols = config.get('mzi_cols', 4)\n            verilog_code += self.generate_mzi_array(rows, cols, config.get('precision', 8))\n            \n        # Accumulator tree\n        if config.get('include_accumulator', True):\n            verilog_code += self.generate_accumulator_tree(config.get('mzi_cols', 4), config.get('precision', 8))\n            \n        # Pipeline registers\n        if config.get('pipeline_stages', 3) > 1:\n            verilog_code += self.generate_pipeline_registers(\n                config.get('pipeline_stages', 3),\n                config.get('output_width', 32)\n            )\n            \n        # Module footer\n        verilog_code += \"endmodule\\n\"\n        \n        # Add generation timestamp\n        timestamp = datetime.now().strftime(\"%Y-%m-%d %H:%M:%S\")\n        header_comment = f\"\"\"//\n// Generated by PhotonicFoundry Verilog Generator\n// Timestamp: {timestamp}\n// Target PDK: {self.target_pdk}\n// Module: {module_name}\n//\n\n\"\"\"\n        \n        return header_comment + verilog_code\n\n\nclass TestbenchGenerator:\n    \"\"\"Comprehensive testbench generation for photonic circuits.\"\"\"\n    \n    def __init__(self):\n        self.test_vectors = []\n        self.expected_outputs = []\n        \n    def generate_basic_testbench(self, module_name: str, config: Dict[str, Any]) -> str:\n        \"\"\"Generate basic testbench with clock and reset.\"\"\"\n        tb_name = f\"tb_{module_name}\"\n        \n        testbench = f\"\"\"\n// Testbench for {module_name}\n`timescale 1ps/1fs\n\nmodule {tb_name};\n\n// Parameters\nparameter CLK_PERIOD = 1000; // 1ns = 1GHz\nparameter INPUT_WIDTH = {config.get('input_width', 32)};\nparameter OUTPUT_WIDTH = {config.get('output_width', 32)};\n\n// Clock and reset\nreg clk;\nreg rst_n;\n\n// DUT signals\nreg [INPUT_WIDTH-1:0] data_in;\nreg valid_in;\nwire [OUTPUT_WIDTH-1:0] data_out;\nwire valid_out;\n\n// Test control\nreg [31:0] test_count;\nreg [31:0] error_count;\nreg test_complete;\n\n// DUT instantiation\n{module_name} dut (\n    .clk(clk),\n    .rst_n(rst_n),\n    .data_in(data_in),\n    .valid_in(valid_in),\n    .data_out(data_out),\n    .valid_out(valid_out)\n);\n\n// Clock generation\ninitial begin\n    clk = 0;\n    forever #(CLK_PERIOD/2) clk = ~clk;\nend\n\n// Reset sequence\ninitial begin\n    rst_n = 0;\n    #(CLK_PERIOD*10) rst_n = 1;\nend\n\n// Test sequence\ninitial begin\n    // Initialize\n    data_in = 0;\n    valid_in = 0;\n    test_count = 0;\n    error_count = 0;\n    test_complete = 0;\n    \n    // Wait for reset deassertion\n    wait(rst_n);\n    @(posedge clk);\n    \n    // Run test vectors\n    run_test_vectors();\n    \n    // Complete\n    test_complete = 1;\n    #(CLK_PERIOD*10);\n    \n    // Report results\n    $display(\"\\nTest Summary:\");\n    $display(\"Total tests: %d\", test_count);\n    $display(\"Errors: %d\", error_count);\n    if (error_count == 0) begin\n        $display(\"ALL TESTS PASSED!\");\n    end else begin\n        $display(\"TESTS FAILED!\");\n    end\n    \n    $finish;\nend\n\n// Test vector execution\ntask run_test_vectors;\nbegin\n    // Test 1: Basic functionality\n    send_data(32'h12345678);\n    send_data(32'hDEADBEEF);\n    send_data(32'hCAFEBABE);\n    \n    // Test 2: Edge cases\n    send_data(32'h00000000);\n    send_data(32'hFFFFFFFF);\n    \n    // Test 3: Random data\n    repeat(100) begin\n        send_data($random);\n    end\nend\nendtask\n\n// Send data task\ntask send_data(input [INPUT_WIDTH-1:0] data);\nbegin\n    @(posedge clk);\n    data_in = data;\n    valid_in = 1'b1;\n    \n    @(posedge clk);\n    valid_in = 1'b0;\n    \n    // Wait for output\n    wait(valid_out);\n    @(posedge clk);\n    \n    test_count = test_count + 1;\n    $display(\"Test %d: Input=0x%h, Output=0x%h\", test_count, data, data_out);\nend\nendtask\n\n// Waveform dumping\ninitial begin\n    $dumpfile(\"{module_name}_tb.vcd\");\n    $dumpvars(0, {tb_name});\nend\n\n// Timeout watchdog\ninitial begin\n    #(CLK_PERIOD * 100000); // 100k cycles\n    if (!test_complete) begin\n        $display(\"ERROR: Test timeout!\");\n        $finish;\n    end\nend\n\nendmodule\n\"\"\"\n        \n        return testbench\n        \n    def generate_performance_testbench(self, module_name: str, config: Dict[str, Any]) -> str:\n        \"\"\"Generate performance analysis testbench.\"\"\"\n        tb_name = f\"tb_{module_name}_perf\"\n        \n        perf_tb = f\"\"\"\n// Performance testbench for {module_name}\n`timescale 1ps/1fs\n\nmodule {tb_name};\n\n// Performance monitoring\nreal start_time, end_time;\nreal throughput;\ninteger cycle_count;\ninteger data_count;\n\n// Same signals as basic testbench\nreg clk, rst_n;\nreg [31:0] data_in;\nreg valid_in;\nwire [31:0] data_out;\nwire valid_out;\n\n// DUT\n{module_name} dut (\n    .clk(clk),\n    .rst_n(rst_n),\n    .data_in(data_in),\n    .valid_in(valid_in),\n    .data_out(data_out),\n    .valid_out(valid_out)\n);\n\n// Clock generation\ninitial begin\n    clk = 0;\n    forever #500 clk = ~clk; // 1GHz\nend\n\n// Performance test\ninitial begin\n    rst_n = 0;\n    data_in = 0;\n    valid_in = 0;\n    cycle_count = 0;\n    data_count = 0;\n    \n    #1000 rst_n = 1;\n    \n    // Start performance measurement\n    start_time = $realtime;\n    \n    // Send continuous data stream\n    repeat(10000) begin\n        @(posedge clk);\n        data_in = $random;\n        valid_in = 1'b1;\n        cycle_count = cycle_count + 1;\n        \n        if (valid_out) begin\n            data_count = data_count + 1;\n        end\n    end\n    \n    // End measurement\n    end_time = $realtime;\n    \n    // Calculate metrics\n    throughput = data_count / ((end_time - start_time) / 1e12); // Operations per second\n    \n    $display(\"\\nPerformance Results:\");\n    $display(\"Total cycles: %d\", cycle_count);\n    $display(\"Data processed: %d\", data_count);\n    $display(\"Throughput: %.2f GOPS\", throughput / 1e9);\n    $display(\"Latency: %.2f cycles\", real(cycle_count) / real(data_count));\n    \n    $finish;\nend\n\nendmodule\n\"\"\"\n        \n        return perf_tb\n        \n    def generate_coverage_testbench(self, module_name: str, config: Dict[str, Any]) -> str:\n        \"\"\"Generate functional coverage testbench.\"\"\"\n        tb_name = f\"tb_{module_name}_cov\"\n        \n        cov_tb = f\"\"\"\n// Coverage testbench for {module_name}\n`timescale 1ps/1fs\n\nmodule {tb_name};\n\n// Coverage groups\ncovergroup data_values @(posedge clk);\n    data_cp: coverpoint data_in {{\n        bins zero = {{0}};\n        bins low = {{[1:100]}};\n        bins mid = {{[101:65434]}};\n        bins high = {{[65435:$]}};\n        bins max = {{32'hFFFFFFFF}};\n    }}\n    \n    valid_cp: coverpoint valid_in {{\n        bins valid_high = {{1}};\n        bins valid_low = {{0}};\n    }}\n    \n    cross_cp: cross data_cp, valid_cp;\nendgroup\n\ndata_values dv = new();\n\n// Standard testbench signals\nreg clk, rst_n;\nreg [31:0] data_in;\nreg valid_in;\nwire [31:0] data_out;\nwire valid_out;\n\n// DUT\n{module_name} dut (\n    .clk(clk),\n    .rst_n(rst_n),\n    .data_in(data_in),\n    .valid_in(valid_in),\n    .data_out(data_out),\n    .valid_out(valid_out)\n);\n\n// Clock and reset\ninitial begin\n    clk = 0;\n    forever #500 clk = ~clk;\nend\n\ninitial begin\n    rst_n = 0;\n    #1000 rst_n = 1;\nend\n\n// Directed coverage test\ninitial begin\n    data_in = 0;\n    valid_in = 0;\n    \n    wait(rst_n);\n    @(posedge clk);\n    \n    // Cover all bins\n    test_corner_cases();\n    \n    // Wait for coverage to reach 100%\n    wait(dv.get_coverage() >= 100.0);\n    \n    $display(\"\\nCoverage Results:\");\n    $display(\"Functional coverage: %.1f%%\", dv.get_coverage());\n    \n    $finish;\nend\n\ntask test_corner_cases;\nbegin\n    // Test each coverage bin\n    send_and_check(32'h00000000);  // zero\n    send_and_check(32'h00000001);  // low\n    send_and_check(32'h00001000);  // mid\n    send_and_check(32'hFFFFFFFE);  // high\n    send_and_check(32'hFFFFFFFF);  // max\n    \n    // Random testing\n    repeat(1000) begin\n        send_and_check($random);\n    end\nend\nendtask\n\ntask send_and_check(input [31:0] data);\nbegin\n    @(posedge clk);\n    data_in = data;\n    valid_in = 1'b1;\n    \n    @(posedge clk);\n    valid_in = 1'b0;\n    \n    // Wait for response\n    repeat(10) @(posedge clk);\nend\nendtask\n\nendmodule\n\"\"\"\n        \n        return cov_tb