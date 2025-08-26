/*
 * Generated Photonic Neural Network Circuit: simple_mlp
 * PDK: skywater130
 * Operating Wavelength: 1550 nm
 * Total Layers: 3
 * Total Components: 234752
 */

module simple_mlp_photonic (
    input wire clk,
    input wire rst_n,
    input wire [7:0] data_in [783:0],  // 784 input features
    input wire valid_in,
    output wire [7:0] data_out [9:0],  // 10 output classes  
    output wire valid_out
);

// Layer 1: 784 → 256 (MZI Mesh)
wire [7:0] layer1_out [255:0];
wire layer1_valid;

mzi_layer_784x256 layer1_inst (
    .clk(clk),
    .rst_n(rst_n),
    .data_in(data_in),
    .valid_in(valid_in),
    .data_out(layer1_out),
    .valid_out(layer1_valid)
);

// Layer 2: 256 → 128 (MZI Mesh)
wire [7:0] layer2_out [127:0];
wire layer2_valid;

mzi_layer_256x128 layer2_inst (
    .clk(clk),
    .rst_n(rst_n),
    .data_in(layer1_out),
    .valid_in(layer1_valid),
    .data_out(layer2_out),
    .valid_out(layer2_valid)
);

// Layer 3: 128 → 10 (MZI Mesh)
mzi_layer_128x10 layer3_inst (
    .clk(clk),
    .rst_n(rst_n),
    .data_in(layer2_out),
    .valid_in(layer2_valid),
    .data_out(data_out),
    .valid_out(valid_out)
);

endmodule

// MZI-based matrix multiplication unit template
module mzi_unit #(
    parameter PRECISION = 8
) (
    input wire clk,
    input wire rst_n,
    input wire [PRECISION-1:0] data_in,
    input wire [PRECISION-1:0] weight_config,
    output reg [PRECISION-1:0] data_out
);

// Simplified MZI operation: multiplication with phase shift
always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        data_out <= 0;
    end else begin
        // Simulate photonic matrix multiplication
        data_out <= (data_in * weight_config) >> (PRECISION - 4);
    end
end

endmodule