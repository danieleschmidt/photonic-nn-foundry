
// MZI-based linear layer: 4 -> 8
module mzi_layer_4x8 (
    input clk,
    input rst_n,
    input [7:0] data_in [3:0],
    input valid_in,
    output [7:0] data_out [7:0],
    output valid_out
);

// MZI mesh implementation
genvar i, j;
generate
    for (i = 0; i < 8; i = i + 1) begin: row_gen
        for (j = 0; j < 4; j = j + 1) begin: col_gen
            mzi_unit #(
                .PRECISION(8),
                .WEIGHT(8'h80)
            ) mzi_inst (
                .clk(clk),
                .rst_n(rst_n),
                .data_in(data_in[j]),
                .weight_out(intermediate[i][j])
            );
        end
    end
endgenerate

// Accumulation network
reg [7:0] accumulator [7:0];
reg valid_out_reg;

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        valid_out_reg <= 1'b0;
    end else begin
        valid_out_reg <= valid_in;
        for (int k = 0; k < 8; k++) begin
            accumulator[k] <= intermediate[k][0]; // Simplified accumulation
        end
    end
end

assign data_out = accumulator;
assign valid_out = valid_out_reg;

endmodule


// MZI-based linear layer: 8 -> 2
module mzi_layer_8x2 (
    input clk,
    input rst_n,
    input [7:0] data_in [7:0],
    input valid_in,
    output [7:0] data_out [1:0],
    output valid_out
);

// MZI mesh implementation
genvar i, j;
generate
    for (i = 0; i < 2; i = i + 1) begin: row_gen
        for (j = 0; j < 8; j = j + 1) begin: col_gen
            mzi_unit #(
                .PRECISION(8),
                .WEIGHT(8'h80)
            ) mzi_inst (
                .clk(clk),
                .rst_n(rst_n),
                .data_in(data_in[j]),
                .weight_out(intermediate[i][j])
            );
        end
    end
endgenerate

// Accumulation network
reg [7:0] accumulator [1:0];
reg valid_out_reg;

always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        valid_out_reg <= 1'b0;
    end else begin
        valid_out_reg <= valid_in;
        for (int k = 0; k < 2; k++) begin
            accumulator[k] <= intermediate[k][0]; // Simplified accumulation
        end
    end
end

assign data_out = accumulator;
assign valid_out = valid_out_reg;

endmodule

// Top-level photonic neural network: simple_model_3layers
module simple_model_3layers_top (
    input clk,
    input rst_n,
    input [7:0] data_in [3:0],
    input valid_in,
    output [7:0] data_out [1:0],
    output valid_out
);

    // Intermediate signals
    wire [7:0] layer_out [1:0];
    wire layer_valid;
    
    // Layer instantiations
    layer_0_inst layer_0 (.clk(clk), .rst_n(rst_n)); layer_1_inst layer_1 (.clk(clk), .rst_n(rst_n)); 
    
endmodule
