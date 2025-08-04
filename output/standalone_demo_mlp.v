//
// Generated Photonic Neural Network: standalone_demo_mlp
// Target PDK: skywater130
// Operating Wavelength: 1550nm
// Total Layers: 3
// Total Components: 56
//

module standalone_demo_mlp (
    input clk,
    input rst_n,
    input [31:0] data_in,
    input valid_in,
    output [31:0] data_out,
    output valid_out
);

// Parameters
parameter INPUT_WIDTH = 32;
parameter OUTPUT_WIDTH = 32;
parameter PRECISION = 8;
parameter NUM_LAYERS = 3;

// Internal signals
wire [PRECISION-1:0] layer_interconnect [NUM_LAYERS:0];
wire layer_valid [NUM_LAYERS:0];

// Input assignment
assign layer_interconnect[0] = data_in[PRECISION-1:0];
assign layer_valid[0] = valid_in;

// Layer instantiations
genvar i;
generate
    for (i = 0; i < NUM_LAYERS; i = i + 1) begin: layer_gen
        photonic_layer #(
            .LAYER_TYPE(i),
            .PRECISION(PRECISION)
        ) layer_inst (
            .clk(clk),
            .rst_n(rst_n),
            .data_in(layer_interconnect[i]),
            .valid_in(layer_valid[i]),
            .data_out(layer_interconnect[i+1]),
            .valid_out(layer_valid[i+1])
        );
    end
endgenerate

// Output assignment
assign data_out = {{(INPUT_WIDTH-PRECISION){1'b0}}, layer_interconnect[NUM_LAYERS]};
assign valid_out = layer_valid[NUM_LAYERS];

endmodule

// Basic photonic layer module
module photonic_layer #(
    parameter LAYER_TYPE = 0,
    parameter PRECISION = 8
) (
    input clk,
    input rst_n,
    input [PRECISION-1:0] data_in,
    input valid_in,
    output [PRECISION-1:0] data_out,
    output valid_out
);

// Processing pipeline
reg [PRECISION-1:0] stage1_data, stage2_data, stage3_data;
reg stage1_valid, stage2_valid, stage3_valid;

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        stage1_data <= 0;
        stage2_data <= 0;
        stage3_data <= 0;
        stage1_valid <= 1'b0;
        stage2_valid <= 1'b0;
        stage3_valid <= 1'b0;
    end else begin
        // Stage 1: Input capture
        stage1_data <= data_in;
        stage1_valid <= valid_in;
        
        // Stage 2: Photonic processing (simplified)
        if (LAYER_TYPE == 0 || LAYER_TYPE == 2) begin
            // Linear layer: matrix multiplication simulation
            stage2_data <= stage1_data + 1; // Simplified transformation
        end else begin
            // Activation layer: ReLU simulation
            stage2_data <= (stage1_data[PRECISION-1]) ? 8'h00 : stage1_data;
        end
        stage2_valid <= stage1_valid;
        
        // Stage 3: Output
        stage3_data <= stage2_data;
        stage3_valid <= stage2_valid;
    end
end

assign data_out = stage3_data;
assign valid_out = stage3_valid;

endmodule
