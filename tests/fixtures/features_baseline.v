module features_before (
    clock,
    clear,
    book$valid,
    book$symbol,
    book$bids,
    book$asks,
    book$flow,
    book$t_first,
    book$t_last,
    features$valid,
    features$value,
    context$symbol,
    context$bid,
    context$has_bid,
    context$ask,
    context$has_ask,
    context$t_first,
    context$t_last
);

    input clock;
    input clear;
    input book$valid;
    input [2:0] book$symbol;
    input [255:0] book$bids;
    input [255:0] book$asks;
    input [19:0] book$flow;
    input [31:0] book$t_first;
    input [31:0] book$t_last;
    output features$valid;
    output [31:0] features$value;
    output [2:0] context$symbol;
    output [31:0] context$bid;
    output context$has_bid;
    output [31:0] context$ask;
    output context$has_ask;
    output [31:0] context$t_first;
    output [31:0] context$t_last;

    wire [31:0] signal_wire;
    reg [31:0] signal_reg;
    reg [31:0] signal_reg_1;
    reg [31:0] signal_reg_2;
    wire [31:0] signal_wire_1;
    reg [31:0] signal_reg_3;
    reg [31:0] signal_reg_4;
    reg [31:0] signal_reg_5;
    reg signal_reg_6;
    reg signal_reg_7;
    reg signal_reg_8;
    reg [31:0] signal_reg_9;
    reg [31:0] signal_reg_10;
    reg [31:0] signal_reg_11;
    reg signal_reg_12;
    reg signal_reg_13;
    reg signal_reg_14;
    reg [31:0] signal_reg_15;
    reg [31:0] signal_reg_16;
    reg [31:0] signal_reg_17;
    wire [2:0] signal_wire_2;
    reg [2:0] signal_reg_18;
    reg [2:0] signal_reg_19;
    reg [2:0] signal_reg_20;
    wire [7:0] signal_select;
    wire [7:0] signal_const;
    wire [31:0] signal_const_1;
    wire [31:0] signal_select_1;
    wire [31:0] signal_select_2;
    wire [31:0] signal_sub;
    wire [25:0] signal_select_3;
    wire [5:0] signal_const_2;
    wire [31:0] signal_cat;
    wire signal_lt;
    wire [31:0] signal_const_3;
    wire signal_eq;
    wire signal_not;
    wire signal_eq_1;
    wire signal_not_1;
    wire signal_and;
    wire signal_and_1;
    wire [7:0] signal_mux;
    reg [7:0] spread;
    reg [7:0] signal_reg_21;
    reg [7:0] signal_reg_22;
    wire [15:0] signal_select_4;
    wire [15:0] signal_const_6;
    wire [31:0] signal_cat_1;
    wire [23:0] signal_select_5;
    wire [7:0] signal_const_7;
    wire [31:0] signal_cat_2;
    wire [27:0] signal_select_6;
    wire [3:0] signal_const_8;
    wire [31:0] signal_cat_3;
    wire [29:0] signal_select_7;
    wire [1:0] signal_const_9;
    wire [31:0] signal_cat_4;
    wire [30:0] signal_select_8;
    wire signal_const_10;
    wire [31:0] signal_cat_5;
    wire signal_select_9;
    wire [31:0] signal_mux_1;
    wire signal_select_10;
    wire [31:0] signal_mux_2;
    wire signal_select_11;
    wire [31:0] signal_mux_3;
    wire signal_select_12;
    wire [31:0] signal_mux_4;
    wire signal_select_13;
    wire [31:0] signal_mux_5;
    wire signal_select_14;
    wire [31:0] signal_mux_6;
    wire [3:0] signal_select_15;
    reg [3:0] signal_reg_23;
    wire [15:0] signal_select_16;
    wire [31:0] signal_cat_6;
    wire [23:0] signal_select_17;
    wire [31:0] signal_cat_7;
    wire [27:0] signal_select_18;
    wire [31:0] signal_cat_8;
    wire [29:0] signal_select_19;
    wire [31:0] signal_cat_9;
    wire [30:0] signal_select_20;
    wire [31:0] signal_cat_10;
    wire signal_select_21;
    wire [31:0] signal_mux_7;
    wire signal_select_22;
    wire [31:0] signal_mux_8;
    wire signal_select_23;
    wire [31:0] signal_mux_9;
    wire signal_select_24;
    wire [31:0] signal_mux_10;
    wire signal_select_25;
    wire [31:0] signal_mux_11;
    wire [5:0] signal_const_18;
    wire [5:0] signal_sub_1;
    wire [5:0] signal_const_21;
    wire [5:0] signal_mux_12;
    wire [5:0] signal_const_22;
    wire [5:0] signal_const_23;
    wire [5:0] signal_mux_13;
    wire [5:0] signal_mux_14;
    wire [5:0] signal_const_24;
    wire [5:0] signal_const_25;
    wire [5:0] signal_mux_15;
    wire [5:0] signal_const_26;
    wire [5:0] signal_const_27;
    wire [5:0] signal_mux_16;
    wire [5:0] signal_mux_17;
    wire [5:0] signal_mux_18;
    wire [5:0] signal_const_28;
    wire [5:0] signal_const_29;
    wire [5:0] signal_mux_19;
    wire [5:0] signal_const_30;
    wire [5:0] signal_const_31;
    wire [5:0] signal_mux_20;
    wire [5:0] signal_mux_21;
    wire [5:0] signal_const_32;
    wire [5:0] signal_const_33;
    wire [5:0] signal_mux_22;
    wire [5:0] signal_const_34;
    wire [5:0] signal_const_35;
    wire [5:0] signal_mux_23;
    wire [5:0] signal_mux_24;
    wire [5:0] signal_mux_25;
    wire [5:0] signal_mux_26;
    wire [5:0] signal_const_36;
    wire [5:0] signal_const_37;
    wire [5:0] signal_mux_27;
    wire [5:0] signal_const_38;
    wire [5:0] signal_const_39;
    wire [5:0] signal_mux_28;
    wire [5:0] signal_mux_29;
    wire [5:0] signal_const_40;
    wire [5:0] signal_const_41;
    wire [5:0] signal_mux_30;
    wire [5:0] signal_const_42;
    wire [5:0] signal_const_43;
    wire [5:0] signal_mux_31;
    wire [5:0] signal_mux_32;
    wire [5:0] signal_mux_33;
    wire [5:0] signal_const_44;
    wire [5:0] signal_const_45;
    wire [5:0] signal_mux_34;
    wire [5:0] signal_const_46;
    wire [5:0] signal_const_47;
    wire [5:0] signal_mux_35;
    wire [5:0] signal_mux_36;
    wire [5:0] signal_const_49;
    wire [5:0] signal_mux_37;
    wire [5:0] signal_const_50;
    wire [5:0] signal_const_51;
    wire [5:0] signal_mux_38;
    wire [5:0] signal_mux_39;
    wire [5:0] signal_mux_40;
    wire [5:0] signal_mux_41;
    wire [5:0] signal_mux_42;
    wire [5:0] signal_const_52;
    wire signal_select_26;
    wire signal_select_27;
    wire signal_or;
    wire signal_select_28;
    wire signal_select_29;
    wire signal_or_1;
    wire signal_or_2;
    wire signal_select_30;
    wire signal_select_31;
    wire signal_or_3;
    wire signal_select_32;
    wire signal_select_33;
    wire signal_or_4;
    wire signal_or_5;
    wire signal_or_6;
    wire signal_select_34;
    wire signal_select_35;
    wire signal_or_7;
    wire signal_select_36;
    wire signal_select_37;
    wire signal_or_8;
    wire signal_or_9;
    wire signal_select_38;
    wire signal_select_39;
    wire signal_or_10;
    wire signal_select_40;
    wire signal_select_41;
    wire signal_or_11;
    wire signal_or_12;
    wire signal_or_13;
    wire signal_or_14;
    wire signal_select_42;
    wire signal_select_43;
    wire signal_or_15;
    wire signal_select_44;
    wire signal_select_45;
    wire signal_or_16;
    wire signal_or_17;
    wire signal_select_46;
    wire signal_select_47;
    wire signal_or_18;
    wire signal_select_48;
    wire signal_select_49;
    wire signal_or_19;
    wire signal_or_20;
    wire signal_or_21;
    wire signal_select_50;
    wire signal_select_51;
    wire signal_or_22;
    wire signal_select_52;
    wire signal_select_53;
    wire signal_or_23;
    wire signal_or_24;
    wire signal_select_54;
    wire signal_select_55;
    wire signal_or_25;
    wire signal_select_56;
    reg [31:0] top_ask;
    reg [31:0] top_bid;
    wire [31:0] signal_or_26;
    wire signal_select_57;
    wire signal_or_27;
    wire signal_or_28;
    wire signal_or_29;
    wire signal_or_30;
    wire signal_or_31;
    wire [5:0] signal_mux_43;
    wire signal_lt_1;
    wire signal_not_2;
    wire [5:0] signal_mux_44;
    wire signal_select_58;
    wire [31:0] signal_mux_45;
    wire [3:0] signal_select_59;
    reg [3:0] signal_reg_24;
    wire [7:0] top_index;
    reg [7:0] signal_mux_46;
    reg [7:0] signal_reg_25;
    wire [7:0] signal_const_54;
    wire [7:0] signal_const_55;
    wire [7:0] signal_const_56;
    wire [7:0] signal_const_57;
    wire [7:0] signal_const_58;
    wire [7:0] signal_const_59;
    wire [7:0] signal_const_60;
    wire [7:0] signal_const_61;
    wire [7:0] signal_const_62;
    wire [7:0] signal_const_63;
    wire [7:0] signal_const_64;
    wire [7:0] signal_const_65;
    wire [7:0] signal_const_66;
    wire [7:0] signal_const_67;
    wire [7:0] signal_const_69;
    wire [7:0] signal_const_71;
    wire [7:0] signal_const_72;
    wire [7:0] signal_const_73;
    wire [7:0] signal_const_74;
    wire [7:0] signal_const_75;
    wire [7:0] signal_const_76;
    wire [7:0] signal_const_77;
    wire [7:0] signal_const_78;
    wire [7:0] signal_const_79;
    wire [7:0] signal_const_80;
    wire [7:0] signal_const_81;
    wire [7:0] signal_const_82;
    wire [7:0] signal_const_83;
    wire [7:0] signal_const_85;
    wire [7:0] signal_const_86;
    wire [7:0] signal_const_89;
    wire [7:0] signal_const_90;
    wire [7:0] signal_const_91;
    wire [7:0] signal_const_92;
    wire [7:0] signal_const_93;
    wire [7:0] signal_const_94;
    wire [7:0] signal_const_95;
    wire [7:0] signal_const_96;
    wire [7:0] signal_const_97;
    wire [7:0] signal_const_98;
    wire [7:0] signal_const_99;
    wire [7:0] signal_const_101;
    wire [7:0] signal_const_102;
    wire [7:0] signal_const_105;
    wire [7:0] signal_const_106;
    wire [7:0] signal_const_107;
    wire [7:0] signal_const_109;
    wire [7:0] signal_const_111;
    wire [7:0] signal_const_113;
    wire [7:0] signal_const_114;
    wire [7:0] signal_const_115;
    wire [7:0] signal_const_117;
    wire [7:0] signal_const_118;
    wire [7:0] signal_const_119;
    wire [7:0] signal_const_120;
    wire [7:0] signal_const_123;
    wire [7:0] signal_const_126;
    wire [7:0] signal_const_127;
    wire [7:0] signal_const_128;
    wire [7:0] signal_const_129;
    wire [7:0] signal_const_130;
    wire [7:0] signal_const_131;
    wire [7:0] signal_const_133;
    wire [7:0] signal_const_134;
    wire [7:0] signal_const_135;
    wire [7:0] signal_const_136;
    wire [7:0] signal_const_139;
    wire [7:0] signal_const_141;
    wire [7:0] signal_const_145;
    wire [7:0] signal_const_147;
    wire [7:0] signal_const_149;
    wire [7:0] signal_const_150;
    wire [7:0] signal_const_151;
    wire [7:0] signal_const_152;
    wire [7:0] signal_const_153;
    wire [7:0] signal_const_154;
    wire [7:0] signal_const_157;
    wire [7:0] signal_const_159;
    wire [7:0] signal_const_160;
    wire [7:0] signal_const_162;
    wire [7:0] signal_const_163;
    wire [7:0] signal_const_165;
    wire [7:0] signal_const_166;
    wire [7:0] signal_const_167;
    wire [7:0] signal_const_173;
    wire [7:0] signal_const_175;
    wire [7:0] signal_const_177;
    wire [7:0] signal_const_179;
    wire [7:0] signal_const_181;
    wire [7:0] signal_const_182;
    wire [7:0] signal_const_183;
    wire [7:0] signal_const_184;
    wire [7:0] signal_const_186;
    wire [7:0] signal_const_187;
    wire [7:0] signal_const_188;
    wire [7:0] signal_const_197;
    wire [7:0] signal_const_198;
    wire [7:0] signal_const_199;
    wire [7:0] signal_const_201;
    wire [7:0] signal_const_213;
    wire [7:0] signal_const_214;
    wire [7:0] signal_const_215;
    wire [7:0] signal_const_216;
    wire [7:0] signal_const_217;
    wire [7:0] signal_const_219;
    wire [7:0] signal_const_220;
    wire [7:0] signal_const_229;
    wire [7:0] signal_const_230;
    wire [7:0] signal_const_231;
    wire [7:0] signal_const_233;
    wire [7:0] signal_const_235;
    wire [7:0] signal_const_245;
    wire [7:0] signal_const_246;
    wire [7:0] signal_const_247;
    wire [7:0] signal_const_248;
    wire [7:0] signal_const_249;
    wire [7:0] signal_const_250;
    wire [7:0] signal_const_252;
    wire [7:0] signal_const_261;
    wire [7:0] signal_const_262;
    wire [7:0] signal_const_263;
    wire [7:0] signal_const_264;
    wire [7:0] signal_const_265;
    wire [7:0] signal_const_267;
    wire [7:0] signal_const_277;
    wire [7:0] signal_const_278;
    wire [7:0] signal_const_279;
    wire [7:0] signal_const_280;
    wire [7:0] signal_const_281;
    wire [7:0] signal_const_282;
    wire [7:0] signal_const_283;
    wire [7:0] signal_const_284;
    wire [7:0] signal_const_293;
    wire [1:0] signal_select_60;
    wire [33:0] signal_cat_11;
    wire [17:0] signal_select_61;
    wire [33:0] signal_cat_12;
    wire [25:0] signal_select_62;
    wire [33:0] signal_cat_13;
    wire [29:0] signal_select_63;
    wire [33:0] signal_cat_14;
    wire [31:0] signal_select_64;
    wire [33:0] signal_cat_15;
    wire [32:0] signal_select_65;
    wire [33:0] signal_cat_16;
    wire signal_select_66;
    wire [33:0] signal_mux_47;
    wire signal_select_67;
    wire [33:0] signal_mux_48;
    wire signal_select_68;
    wire [33:0] signal_mux_49;
    wire signal_select_69;
    wire [33:0] signal_mux_50;
    wire signal_select_70;
    wire [33:0] signal_mux_51;
    wire signal_select_71;
    wire [33:0] signal_mux_52;
    wire [3:0] signal_select_72;
    reg [3:0] signal_reg_26;
    wire [1:0] signal_select_73;
    wire [33:0] signal_cat_17;
    wire [17:0] signal_select_74;
    wire [33:0] signal_cat_18;
    wire [25:0] signal_select_75;
    wire [33:0] signal_cat_19;
    wire [29:0] signal_select_76;
    wire [33:0] signal_cat_20;
    wire [31:0] signal_select_77;
    wire [33:0] signal_cat_21;
    wire [32:0] signal_select_78;
    wire [33:0] signal_cat_22;
    wire signal_select_79;
    wire [33:0] signal_mux_53;
    wire signal_select_80;
    wire [33:0] signal_mux_54;
    wire signal_select_81;
    wire [33:0] signal_mux_55;
    wire signal_select_82;
    wire [33:0] signal_mux_56;
    wire signal_select_83;
    wire [33:0] signal_mux_57;
    wire [5:0] signal_sub_2;
    wire [5:0] signal_mux_58;
    wire [5:0] signal_mux_59;
    wire [5:0] signal_mux_60;
    wire [5:0] signal_mux_61;
    wire [5:0] signal_mux_62;
    wire [5:0] signal_mux_63;
    wire [5:0] signal_mux_64;
    wire [5:0] signal_mux_65;
    wire [5:0] signal_mux_66;
    wire [5:0] signal_mux_67;
    wire [5:0] signal_mux_68;
    wire [5:0] signal_mux_69;
    wire [5:0] signal_mux_70;
    wire [5:0] signal_mux_71;
    wire [5:0] signal_mux_72;
    wire [5:0] signal_mux_73;
    wire [5:0] signal_mux_74;
    wire [5:0] signal_mux_75;
    wire [5:0] signal_mux_76;
    wire [5:0] signal_mux_77;
    wire [5:0] signal_mux_78;
    wire [5:0] signal_mux_79;
    wire [5:0] signal_mux_80;
    wire [5:0] signal_mux_81;
    wire [5:0] signal_mux_82;
    wire [5:0] signal_mux_83;
    wire [5:0] signal_mux_84;
    wire [5:0] signal_mux_85;
    wire [5:0] signal_mux_86;
    wire [5:0] signal_mux_87;
    wire [5:0] signal_mux_88;
    wire [5:0] signal_const_357;
    wire [5:0] signal_mux_89;
    wire [5:0] signal_mux_90;
    wire [5:0] signal_const_358;
    wire signal_select_84;
    wire signal_select_85;
    wire signal_or_32;
    wire signal_select_86;
    wire signal_select_87;
    wire signal_or_33;
    wire signal_select_88;
    wire signal_select_89;
    wire signal_or_34;
    wire signal_or_35;
    wire signal_select_90;
    wire signal_select_91;
    wire signal_or_36;
    wire signal_select_92;
    wire signal_select_93;
    wire signal_or_37;
    wire signal_or_38;
    wire signal_or_39;
    wire signal_select_94;
    wire signal_select_95;
    wire signal_or_40;
    wire signal_select_96;
    wire signal_select_97;
    wire signal_or_41;
    wire signal_or_42;
    wire signal_select_98;
    wire signal_select_99;
    wire signal_or_43;
    wire signal_select_100;
    wire signal_select_101;
    wire signal_or_44;
    wire signal_or_45;
    wire signal_or_46;
    wire signal_or_47;
    wire signal_select_102;
    wire signal_select_103;
    wire signal_or_48;
    wire signal_select_104;
    wire signal_select_105;
    wire signal_or_49;
    wire signal_or_50;
    wire signal_select_106;
    wire signal_select_107;
    wire signal_or_51;
    wire signal_select_108;
    wire signal_select_109;
    wire signal_or_52;
    wire signal_or_53;
    wire signal_or_54;
    wire signal_select_110;
    wire signal_select_111;
    wire signal_or_55;
    wire signal_select_112;
    wire signal_select_113;
    wire signal_or_56;
    wire signal_or_57;
    wire signal_select_114;
    wire signal_select_115;
    wire signal_or_58;
    wire signal_select_116;
    wire [63:0] signal_select_117;
    wire [31:0] signal_select_118;
    wire [33:0] signal_cat_23;
    wire [127:0] signal_select_119;
    wire [63:0] signal_select_120;
    wire [31:0] signal_select_121;
    wire [33:0] signal_cat_24;
    wire [191:0] signal_select_122;
    wire [63:0] signal_select_123;
    wire [31:0] signal_select_124;
    wire [33:0] signal_cat_25;
    wire [255:0] signal_wire_3;
    wire [63:0] signal_select_125;
    wire [31:0] signal_select_126;
    wire [33:0] signal_cat_26;
    wire [33:0] signal_add;
    wire [33:0] signal_add_1;
    wire [33:0] signal_add_2;
    reg [33:0] depth_ask;
    wire [63:0] signal_select_127;
    wire [31:0] signal_select_128;
    wire [33:0] signal_cat_27;
    wire [127:0] signal_select_129;
    wire [63:0] signal_select_130;
    wire [31:0] signal_select_131;
    wire [33:0] signal_cat_28;
    wire [191:0] signal_select_132;
    wire [63:0] signal_select_133;
    wire [31:0] signal_select_134;
    wire [33:0] signal_cat_29;
    wire [255:0] signal_wire_4;
    wire [63:0] signal_select_135;
    wire [31:0] signal_select_136;
    wire [33:0] signal_cat_30;
    wire [33:0] signal_add_3;
    wire [33:0] signal_add_4;
    wire [33:0] signal_add_5;
    reg [33:0] depth_bid;
    wire [33:0] signal_or_59;
    wire signal_select_137;
    wire signal_or_60;
    wire signal_or_61;
    wire signal_or_62;
    wire signal_or_63;
    wire signal_or_64;
    wire signal_or_65;
    wire [5:0] signal_mux_91;
    wire signal_lt_2;
    wire signal_not_3;
    wire [5:0] signal_mux_92;
    wire signal_select_138;
    wire [33:0] signal_mux_93;
    wire [3:0] signal_select_139;
    reg [3:0] signal_reg_27;
    wire [7:0] depth_index;
    reg [7:0] signal_mux_94;
    reg [7:0] signal_reg_28;
    wire [7:0] signal_select_140;
    wire [19:0] signal_const_369;
    wire [18:0] signal_select_141;
    wire signal_select_142;
    wire signal_not_4;
    wire [19:0] signal_cat_31;
    wire signal_lt_3;
    wire [7:0] signal_mux_95;
    wire [18:0] signal_select_143;
    wire [14:0] signal_select_144;
    wire [19:0] signal_wire_5;
    wire signal_select_145;
    wire [1:0] signal_cat_32;
    wire [3:0] signal_cat_33;
    wire [4:0] signal_cat_34;
    wire [19:0] signal_cat_35;
    wire signal_select_146;
    wire signal_not_5;
    wire [19:0] signal_cat_36;
    wire [19:0] signal_const_370;
    wire signal_lt_4;
    wire [7:0] signal_mux_96;
    reg [7:0] flow;
    reg [7:0] signal_reg_29;
    reg [7:0] signal_reg_30;
    wire [31:0] signal_cat_37;
    wire signal_wire_6;
    wire signal_wire_7;
    wire signal_wire_8;
    reg signal_reg_31;
    reg signal_reg_32;
    reg signal_reg_33;
    assign signal_wire = book$t_last;
    always @(posedge signal_wire_7) begin
        signal_reg <= signal_wire;
    end
    always @(posedge signal_wire_7) begin
        signal_reg_1 <= signal_reg;
    end
    always @(posedge signal_wire_7) begin
        signal_reg_2 <= signal_reg_1;
    end
    assign signal_wire_1 = book$t_first;
    always @(posedge signal_wire_7) begin
        signal_reg_3 <= signal_wire_1;
    end
    always @(posedge signal_wire_7) begin
        signal_reg_4 <= signal_reg_3;
    end
    always @(posedge signal_wire_7) begin
        signal_reg_5 <= signal_reg_4;
    end
    always @(posedge signal_wire_7) begin
        signal_reg_6 <= signal_not;
    end
    always @(posedge signal_wire_7) begin
        signal_reg_7 <= signal_reg_6;
    end
    always @(posedge signal_wire_7) begin
        signal_reg_8 <= signal_reg_7;
    end
    always @(posedge signal_wire_7) begin
        signal_reg_9 <= signal_select_2;
    end
    always @(posedge signal_wire_7) begin
        signal_reg_10 <= signal_reg_9;
    end
    always @(posedge signal_wire_7) begin
        signal_reg_11 <= signal_reg_10;
    end
    always @(posedge signal_wire_7) begin
        signal_reg_12 <= signal_not_1;
    end
    always @(posedge signal_wire_7) begin
        signal_reg_13 <= signal_reg_12;
    end
    always @(posedge signal_wire_7) begin
        signal_reg_14 <= signal_reg_13;
    end
    always @(posedge signal_wire_7) begin
        signal_reg_15 <= signal_select_1;
    end
    always @(posedge signal_wire_7) begin
        signal_reg_16 <= signal_reg_15;
    end
    always @(posedge signal_wire_7) begin
        signal_reg_17 <= signal_reg_16;
    end
    assign signal_wire_2 = book$symbol;
    always @(posedge signal_wire_7) begin
        signal_reg_18 <= signal_wire_2;
    end
    always @(posedge signal_wire_7) begin
        signal_reg_19 <= signal_reg_18;
    end
    always @(posedge signal_wire_7) begin
        signal_reg_20 <= signal_reg_19;
    end
    assign signal_select = signal_cat[7:0];
    assign signal_const = 8'b01111111;
    assign signal_const_1 = 32'b00000000000000000000000001111111;
    assign signal_select_1 = signal_select_135[31:0];
    assign signal_select_2 = signal_select_125[31:0];
    assign signal_sub = signal_select_2 - signal_select_1;
    assign signal_select_3 = signal_sub[31:6];
    assign signal_const_2 = 6'b000000;
    assign signal_cat = { signal_const_2,
                          signal_select_3 };
    assign signal_lt = signal_cat < signal_const_1;
    assign signal_const_3 = 32'b00000000000000000000000000000000;
    assign signal_eq = signal_select_126 == signal_const_3;
    assign signal_not = ~ signal_eq;
    assign signal_eq_1 = signal_select_136 == signal_const_3;
    assign signal_not_1 = ~ signal_eq_1;
    assign signal_and = signal_not_1 & signal_not;
    assign signal_and_1 = signal_and & signal_lt;
    assign signal_mux = signal_and_1 ? signal_select : signal_const;
    always @(posedge signal_wire_7) begin
        spread <= signal_mux;
    end
    always @(posedge signal_wire_7) begin
        signal_reg_21 <= spread;
    end
    always @(posedge signal_wire_7) begin
        signal_reg_22 <= signal_reg_21;
    end
    assign signal_select_4 = signal_mux_4[31:16];
    assign signal_const_6 = 16'b0000000000000000;
    assign signal_cat_1 = { signal_const_6,
                            signal_select_4 };
    assign signal_select_5 = signal_mux_3[31:8];
    assign signal_const_7 = 8'b00000000;
    assign signal_cat_2 = { signal_const_7,
                            signal_select_5 };
    assign signal_select_6 = signal_mux_2[31:4];
    assign signal_const_8 = 4'b0000;
    assign signal_cat_3 = { signal_const_8,
                            signal_select_6 };
    assign signal_select_7 = signal_mux_1[31:2];
    assign signal_const_9 = 2'b00;
    assign signal_cat_4 = { signal_const_9,
                            signal_select_7 };
    assign signal_select_8 = top_ask[31:1];
    assign signal_const_10 = 1'b0;
    assign signal_cat_5 = { signal_const_10,
                            signal_select_8 };
    assign signal_select_9 = signal_mux_44[0:0];
    assign signal_mux_1 = signal_select_9 ? signal_cat_5 : top_ask;
    assign signal_select_10 = signal_mux_44[1:1];
    assign signal_mux_2 = signal_select_10 ? signal_cat_4 : signal_mux_1;
    assign signal_select_11 = signal_mux_44[2:2];
    assign signal_mux_3 = signal_select_11 ? signal_cat_3 : signal_mux_2;
    assign signal_select_12 = signal_mux_44[3:3];
    assign signal_mux_4 = signal_select_12 ? signal_cat_2 : signal_mux_3;
    assign signal_select_13 = signal_mux_44[4:4];
    assign signal_mux_5 = signal_select_13 ? signal_cat_1 : signal_mux_4;
    assign signal_select_14 = signal_mux_44[5:5];
    assign signal_mux_6 = signal_select_14 ? signal_const_3 : signal_mux_5;
    assign signal_select_15 = signal_mux_6[3:0];
    always @(posedge signal_wire_7) begin
        signal_reg_23 <= signal_select_15;
    end
    assign signal_select_16 = signal_mux_10[31:16];
    assign signal_cat_6 = { signal_const_6,
                            signal_select_16 };
    assign signal_select_17 = signal_mux_9[31:8];
    assign signal_cat_7 = { signal_const_7,
                            signal_select_17 };
    assign signal_select_18 = signal_mux_8[31:4];
    assign signal_cat_8 = { signal_const_8,
                            signal_select_18 };
    assign signal_select_19 = signal_mux_7[31:2];
    assign signal_cat_9 = { signal_const_9,
                            signal_select_19 };
    assign signal_select_20 = top_bid[31:1];
    assign signal_cat_10 = { signal_const_10,
                             signal_select_20 };
    assign signal_select_21 = signal_mux_44[0:0];
    assign signal_mux_7 = signal_select_21 ? signal_cat_10 : top_bid;
    assign signal_select_22 = signal_mux_44[1:1];
    assign signal_mux_8 = signal_select_22 ? signal_cat_9 : signal_mux_7;
    assign signal_select_23 = signal_mux_44[2:2];
    assign signal_mux_9 = signal_select_23 ? signal_cat_8 : signal_mux_8;
    assign signal_select_24 = signal_mux_44[3:3];
    assign signal_mux_10 = signal_select_24 ? signal_cat_7 : signal_mux_9;
    assign signal_select_25 = signal_mux_44[4:4];
    assign signal_mux_11 = signal_select_25 ? signal_cat_6 : signal_mux_10;
    assign signal_const_18 = 6'b011100;
    assign signal_sub_1 = signal_const_18 - signal_mux_43;
    assign signal_const_21 = 6'b000001;
    assign signal_mux_12 = signal_select_57 ? signal_const_2 : signal_const_21;
    assign signal_const_22 = 6'b000010;
    assign signal_const_23 = 6'b000011;
    assign signal_mux_13 = signal_select_55 ? signal_const_22 : signal_const_23;
    assign signal_mux_14 = signal_or_27 ? signal_mux_12 : signal_mux_13;
    assign signal_const_24 = 6'b000100;
    assign signal_const_25 = 6'b000101;
    assign signal_mux_15 = signal_select_53 ? signal_const_24 : signal_const_25;
    assign signal_const_26 = 6'b000110;
    assign signal_const_27 = 6'b000111;
    assign signal_mux_16 = signal_select_51 ? signal_const_26 : signal_const_27;
    assign signal_mux_17 = signal_or_23 ? signal_mux_15 : signal_mux_16;
    assign signal_mux_18 = signal_or_28 ? signal_mux_14 : signal_mux_17;
    assign signal_const_28 = 6'b001000;
    assign signal_const_29 = 6'b001001;
    assign signal_mux_19 = signal_select_49 ? signal_const_28 : signal_const_29;
    assign signal_const_30 = 6'b001010;
    assign signal_const_31 = 6'b001011;
    assign signal_mux_20 = signal_select_47 ? signal_const_30 : signal_const_31;
    assign signal_mux_21 = signal_or_19 ? signal_mux_19 : signal_mux_20;
    assign signal_const_32 = 6'b001100;
    assign signal_const_33 = 6'b001101;
    assign signal_mux_22 = signal_select_45 ? signal_const_32 : signal_const_33;
    assign signal_const_34 = 6'b001110;
    assign signal_const_35 = 6'b001111;
    assign signal_mux_23 = signal_select_43 ? signal_const_34 : signal_const_35;
    assign signal_mux_24 = signal_or_16 ? signal_mux_22 : signal_mux_23;
    assign signal_mux_25 = signal_or_20 ? signal_mux_21 : signal_mux_24;
    assign signal_mux_26 = signal_or_29 ? signal_mux_18 : signal_mux_25;
    assign signal_const_36 = 6'b010000;
    assign signal_const_37 = 6'b010001;
    assign signal_mux_27 = signal_select_41 ? signal_const_36 : signal_const_37;
    assign signal_const_38 = 6'b010010;
    assign signal_const_39 = 6'b010011;
    assign signal_mux_28 = signal_select_39 ? signal_const_38 : signal_const_39;
    assign signal_mux_29 = signal_or_11 ? signal_mux_27 : signal_mux_28;
    assign signal_const_40 = 6'b010100;
    assign signal_const_41 = 6'b010101;
    assign signal_mux_30 = signal_select_37 ? signal_const_40 : signal_const_41;
    assign signal_const_42 = 6'b010110;
    assign signal_const_43 = 6'b010111;
    assign signal_mux_31 = signal_select_35 ? signal_const_42 : signal_const_43;
    assign signal_mux_32 = signal_or_8 ? signal_mux_30 : signal_mux_31;
    assign signal_mux_33 = signal_or_12 ? signal_mux_29 : signal_mux_32;
    assign signal_const_44 = 6'b011000;
    assign signal_const_45 = 6'b011001;
    assign signal_mux_34 = signal_select_33 ? signal_const_44 : signal_const_45;
    assign signal_const_46 = 6'b011010;
    assign signal_const_47 = 6'b011011;
    assign signal_mux_35 = signal_select_31 ? signal_const_46 : signal_const_47;
    assign signal_mux_36 = signal_or_4 ? signal_mux_34 : signal_mux_35;
    assign signal_const_49 = 6'b011101;
    assign signal_mux_37 = signal_select_29 ? signal_const_18 : signal_const_49;
    assign signal_const_50 = 6'b011110;
    assign signal_const_51 = 6'b011111;
    assign signal_mux_38 = signal_select_27 ? signal_const_50 : signal_const_51;
    assign signal_mux_39 = signal_or_1 ? signal_mux_37 : signal_mux_38;
    assign signal_mux_40 = signal_or_5 ? signal_mux_36 : signal_mux_39;
    assign signal_mux_41 = signal_or_13 ? signal_mux_33 : signal_mux_40;
    assign signal_mux_42 = signal_or_30 ? signal_mux_26 : signal_mux_41;
    assign signal_const_52 = 6'b100000;
    assign signal_select_26 = signal_or_26[0:0];
    assign signal_select_27 = signal_or_26[1:1];
    assign signal_or = signal_select_27 | signal_select_26;
    assign signal_select_28 = signal_or_26[2:2];
    assign signal_select_29 = signal_or_26[3:3];
    assign signal_or_1 = signal_select_29 | signal_select_28;
    assign signal_or_2 = signal_or_1 | signal_or;
    assign signal_select_30 = signal_or_26[4:4];
    assign signal_select_31 = signal_or_26[5:5];
    assign signal_or_3 = signal_select_31 | signal_select_30;
    assign signal_select_32 = signal_or_26[6:6];
    assign signal_select_33 = signal_or_26[7:7];
    assign signal_or_4 = signal_select_33 | signal_select_32;
    assign signal_or_5 = signal_or_4 | signal_or_3;
    assign signal_or_6 = signal_or_5 | signal_or_2;
    assign signal_select_34 = signal_or_26[8:8];
    assign signal_select_35 = signal_or_26[9:9];
    assign signal_or_7 = signal_select_35 | signal_select_34;
    assign signal_select_36 = signal_or_26[10:10];
    assign signal_select_37 = signal_or_26[11:11];
    assign signal_or_8 = signal_select_37 | signal_select_36;
    assign signal_or_9 = signal_or_8 | signal_or_7;
    assign signal_select_38 = signal_or_26[12:12];
    assign signal_select_39 = signal_or_26[13:13];
    assign signal_or_10 = signal_select_39 | signal_select_38;
    assign signal_select_40 = signal_or_26[14:14];
    assign signal_select_41 = signal_or_26[15:15];
    assign signal_or_11 = signal_select_41 | signal_select_40;
    assign signal_or_12 = signal_or_11 | signal_or_10;
    assign signal_or_13 = signal_or_12 | signal_or_9;
    assign signal_or_14 = signal_or_13 | signal_or_6;
    assign signal_select_42 = signal_or_26[16:16];
    assign signal_select_43 = signal_or_26[17:17];
    assign signal_or_15 = signal_select_43 | signal_select_42;
    assign signal_select_44 = signal_or_26[18:18];
    assign signal_select_45 = signal_or_26[19:19];
    assign signal_or_16 = signal_select_45 | signal_select_44;
    assign signal_or_17 = signal_or_16 | signal_or_15;
    assign signal_select_46 = signal_or_26[20:20];
    assign signal_select_47 = signal_or_26[21:21];
    assign signal_or_18 = signal_select_47 | signal_select_46;
    assign signal_select_48 = signal_or_26[22:22];
    assign signal_select_49 = signal_or_26[23:23];
    assign signal_or_19 = signal_select_49 | signal_select_48;
    assign signal_or_20 = signal_or_19 | signal_or_18;
    assign signal_or_21 = signal_or_20 | signal_or_17;
    assign signal_select_50 = signal_or_26[24:24];
    assign signal_select_51 = signal_or_26[25:25];
    assign signal_or_22 = signal_select_51 | signal_select_50;
    assign signal_select_52 = signal_or_26[26:26];
    assign signal_select_53 = signal_or_26[27:27];
    assign signal_or_23 = signal_select_53 | signal_select_52;
    assign signal_or_24 = signal_or_23 | signal_or_22;
    assign signal_select_54 = signal_or_26[28:28];
    assign signal_select_55 = signal_or_26[29:29];
    assign signal_or_25 = signal_select_55 | signal_select_54;
    assign signal_select_56 = signal_or_26[30:30];
    always @(posedge signal_wire_7) begin
        top_ask <= signal_select_126;
    end
    always @(posedge signal_wire_7) begin
        top_bid <= signal_select_136;
    end
    assign signal_or_26 = top_bid | top_ask;
    assign signal_select_57 = signal_or_26[31:31];
    assign signal_or_27 = signal_select_57 | signal_select_56;
    assign signal_or_28 = signal_or_27 | signal_or_25;
    assign signal_or_29 = signal_or_28 | signal_or_24;
    assign signal_or_30 = signal_or_29 | signal_or_21;
    assign signal_or_31 = signal_or_30 | signal_or_14;
    assign signal_mux_43 = signal_or_31 ? signal_mux_42 : signal_const_52;
    assign signal_lt_1 = signal_mux_43 < signal_const_18;
    assign signal_not_2 = ~ signal_lt_1;
    assign signal_mux_44 = signal_not_2 ? signal_const_2 : signal_sub_1;
    assign signal_select_58 = signal_mux_44[5:5];
    assign signal_mux_45 = signal_select_58 ? signal_const_3 : signal_mux_11;
    assign signal_select_59 = signal_mux_45[3:0];
    always @(posedge signal_wire_7) begin
        signal_reg_24 <= signal_select_59;
    end
    assign top_index = { signal_reg_24,
                         signal_reg_23 };
    always @* begin
        case (top_index)
        0:
            signal_mux_46 <= signal_const_7;
        1:
            signal_mux_46 <= signal_const_293;
        2:
            signal_mux_46 <= signal_const_293;
        3:
            signal_mux_46 <= signal_const_293;
        4:
            signal_mux_46 <= signal_const_293;
        5:
            signal_mux_46 <= signal_const_293;
        6:
            signal_mux_46 <= signal_const_293;
        7:
            signal_mux_46 <= signal_const_293;
        8:
            signal_mux_46 <= signal_const_293;
        9:
            signal_mux_46 <= signal_const_293;
        10:
            signal_mux_46 <= signal_const_293;
        11:
            signal_mux_46 <= signal_const_293;
        12:
            signal_mux_46 <= signal_const_293;
        13:
            signal_mux_46 <= signal_const_293;
        14:
            signal_mux_46 <= signal_const_293;
        15:
            signal_mux_46 <= signal_const_293;
        16:
            signal_mux_46 <= signal_const;
        17:
            signal_mux_46 <= signal_const_7;
        18:
            signal_mux_46 <= signal_const_182;
        19:
            signal_mux_46 <= signal_const_213;
        20:
            signal_mux_46 <= signal_const_248;
        21:
            signal_mux_46 <= signal_const_245;
        22:
            signal_mux_46 <= signal_const_264;
        23:
            signal_mux_46 <= signal_const_262;
        24:
            signal_mux_46 <= signal_const_284;
        25:
            signal_mux_46 <= signal_const_283;
        26:
            signal_mux_46 <= signal_const_282;
        27:
            signal_mux_46 <= signal_const_281;
        28:
            signal_mux_46 <= signal_const_280;
        29:
            signal_mux_46 <= signal_const_279;
        30:
            signal_mux_46 <= signal_const_278;
        31:
            signal_mux_46 <= signal_const_277;
        32:
            signal_mux_46 <= signal_const;
        33:
            signal_mux_46 <= signal_const_77;
        34:
            signal_mux_46 <= signal_const_7;
        35:
            signal_mux_46 <= signal_const_133;
        36:
            signal_mux_46 <= signal_const_182;
        37:
            signal_mux_46 <= signal_const_197;
        38:
            signal_mux_46 <= signal_const_213;
        39:
            signal_mux_46 <= signal_const_230;
        40:
            signal_mux_46 <= signal_const_248;
        41:
            signal_mux_46 <= signal_const_267;
        42:
            signal_mux_46 <= signal_const_245;
        43:
            signal_mux_46 <= signal_const_265;
        44:
            signal_mux_46 <= signal_const_264;
        45:
            signal_mux_46 <= signal_const_263;
        46:
            signal_mux_46 <= signal_const_262;
        47:
            signal_mux_46 <= signal_const_261;
        48:
            signal_mux_46 <= signal_const;
        49:
            signal_mux_46 <= signal_const_63;
        50:
            signal_mux_46 <= signal_const_58;
        51:
            signal_mux_46 <= signal_const_7;
        52:
            signal_mux_46 <= signal_const_152;
        53:
            signal_mux_46 <= signal_const_149;
        54:
            signal_mux_46 <= signal_const_182;
        55:
            signal_mux_46 <= signal_const_198;
        56:
            signal_mux_46 <= signal_const_252;
        57:
            signal_mux_46 <= signal_const_213;
        58:
            signal_mux_46 <= signal_const_250;
        59:
            signal_mux_46 <= signal_const_249;
        60:
            signal_mux_46 <= signal_const_248;
        61:
            signal_mux_46 <= signal_const_247;
        62:
            signal_mux_46 <= signal_const_246;
        63:
            signal_mux_46 <= signal_const_245;
        64:
            signal_mux_46 <= signal_const;
        65:
            signal_mux_46 <= signal_const_113;
        66:
            signal_mux_46 <= signal_const_77;
        67:
            signal_mux_46 <= signal_const_107;
        68:
            signal_mux_46 <= signal_const_7;
        69:
            signal_mux_46 <= signal_const_101;
        70:
            signal_mux_46 <= signal_const_133;
        71:
            signal_mux_46 <= signal_const_166;
        72:
            signal_mux_46 <= signal_const_182;
        73:
            signal_mux_46 <= signal_const_235;
        74:
            signal_mux_46 <= signal_const_197;
        75:
            signal_mux_46 <= signal_const_233;
        76:
            signal_mux_46 <= signal_const_213;
        77:
            signal_mux_46 <= signal_const_231;
        78:
            signal_mux_46 <= signal_const_230;
        79:
            signal_mux_46 <= signal_const_229;
        80:
            signal_mux_46 <= signal_const;
        81:
            signal_mux_46 <= signal_const_65;
        82:
            signal_mux_46 <= signal_const_62;
        83:
            signal_mux_46 <= signal_const_59;
        84:
            signal_mux_46 <= signal_const_56;
        85:
            signal_mux_46 <= signal_const_7;
        86:
            signal_mux_46 <= signal_const_136;
        87:
            signal_mux_46 <= signal_const_134;
        88:
            signal_mux_46 <= signal_const_220;
        89:
            signal_mux_46 <= signal_const_219;
        90:
            signal_mux_46 <= signal_const_182;
        91:
            signal_mux_46 <= signal_const_217;
        92:
            signal_mux_46 <= signal_const_216;
        93:
            signal_mux_46 <= signal_const_215;
        94:
            signal_mux_46 <= signal_const_214;
        95:
            signal_mux_46 <= signal_const_213;
        96:
            signal_mux_46 <= signal_const;
        97:
            signal_mux_46 <= signal_const_114;
        98:
            signal_mux_46 <= signal_const_63;
        99:
            signal_mux_46 <= signal_const_77;
        100:
            signal_mux_46 <= signal_const_58;
        101:
            signal_mux_46 <= signal_const_106;
        102:
            signal_mux_46 <= signal_const_7;
        103:
            signal_mux_46 <= signal_const_102;
        104:
            signal_mux_46 <= signal_const_152;
        105:
            signal_mux_46 <= signal_const_133;
        106:
            signal_mux_46 <= signal_const_149;
        107:
            signal_mux_46 <= signal_const_201;
        108:
            signal_mux_46 <= signal_const_182;
        109:
            signal_mux_46 <= signal_const_199;
        110:
            signal_mux_46 <= signal_const_198;
        111:
            signal_mux_46 <= signal_const_197;
        112:
            signal_mux_46 <= signal_const;
        113:
            signal_mux_46 <= signal_const_82;
        114:
            signal_mux_46 <= signal_const_80;
        115:
            signal_mux_46 <= signal_const_78;
        116:
            signal_mux_46 <= signal_const_76;
        117:
            signal_mux_46 <= signal_const_74;
        118:
            signal_mux_46 <= signal_const_72;
        119:
            signal_mux_46 <= signal_const_7;
        120:
            signal_mux_46 <= signal_const_188;
        121:
            signal_mux_46 <= signal_const_187;
        122:
            signal_mux_46 <= signal_const_186;
        123:
            signal_mux_46 <= signal_const_150;
        124:
            signal_mux_46 <= signal_const_184;
        125:
            signal_mux_46 <= signal_const_183;
        126:
            signal_mux_46 <= signal_const_182;
        127:
            signal_mux_46 <= signal_const_181;
        128:
            signal_mux_46 <= signal_const;
        129:
            signal_mux_46 <= signal_const_179;
        130:
            signal_mux_46 <= signal_const_113;
        131:
            signal_mux_46 <= signal_const_177;
        132:
            signal_mux_46 <= signal_const_77;
        133:
            signal_mux_46 <= signal_const_175;
        134:
            signal_mux_46 <= signal_const_107;
        135:
            signal_mux_46 <= signal_const_173;
        136:
            signal_mux_46 <= signal_const_7;
        137:
            signal_mux_46 <= signal_const_154;
        138:
            signal_mux_46 <= signal_const_101;
        139:
            signal_mux_46 <= signal_const_117;
        140:
            signal_mux_46 <= signal_const_133;
        141:
            signal_mux_46 <= signal_const_167;
        142:
            signal_mux_46 <= signal_const_166;
        143:
            signal_mux_46 <= signal_const_165;
        144:
            signal_mux_46 <= signal_const;
        145:
            signal_mux_46 <= signal_const_163;
        146:
            signal_mux_46 <= signal_const_162;
        147:
            signal_mux_46 <= signal_const_63;
        148:
            signal_mux_46 <= signal_const_160;
        149:
            signal_mux_46 <= signal_const_159;
        150:
            signal_mux_46 <= signal_const_58;
        151:
            signal_mux_46 <= signal_const_157;
        152:
            signal_mux_46 <= signal_const_139;
        153:
            signal_mux_46 <= signal_const_7;
        154:
            signal_mux_46 <= signal_const_154;
        155:
            signal_mux_46 <= signal_const_153;
        156:
            signal_mux_46 <= signal_const_152;
        157:
            signal_mux_46 <= signal_const_151;
        158:
            signal_mux_46 <= signal_const_150;
        159:
            signal_mux_46 <= signal_const_149;
        160:
            signal_mux_46 <= signal_const;
        161:
            signal_mux_46 <= signal_const_147;
        162:
            signal_mux_46 <= signal_const_65;
        163:
            signal_mux_46 <= signal_const_145;
        164:
            signal_mux_46 <= signal_const_62;
        165:
            signal_mux_46 <= signal_const_77;
        166:
            signal_mux_46 <= signal_const_59;
        167:
            signal_mux_46 <= signal_const_141;
        168:
            signal_mux_46 <= signal_const_56;
        169:
            signal_mux_46 <= signal_const_139;
        170:
            signal_mux_46 <= signal_const_7;
        171:
            signal_mux_46 <= signal_const_120;
        172:
            signal_mux_46 <= signal_const_136;
        173:
            signal_mux_46 <= signal_const_135;
        174:
            signal_mux_46 <= signal_const_134;
        175:
            signal_mux_46 <= signal_const_133;
        176:
            signal_mux_46 <= signal_const;
        177:
            signal_mux_46 <= signal_const_131;
        178:
            signal_mux_46 <= signal_const_130;
        179:
            signal_mux_46 <= signal_const_129;
        180:
            signal_mux_46 <= signal_const_128;
        181:
            signal_mux_46 <= signal_const_127;
        182:
            signal_mux_46 <= signal_const_126;
        183:
            signal_mux_46 <= signal_const_75;
        184:
            signal_mux_46 <= signal_const_57;
        185:
            signal_mux_46 <= signal_const_123;
        186:
            signal_mux_46 <= signal_const_105;
        187:
            signal_mux_46 <= signal_const_7;
        188:
            signal_mux_46 <= signal_const_120;
        189:
            signal_mux_46 <= signal_const_119;
        190:
            signal_mux_46 <= signal_const_118;
        191:
            signal_mux_46 <= signal_const_117;
        192:
            signal_mux_46 <= signal_const;
        193:
            signal_mux_46 <= signal_const_115;
        194:
            signal_mux_46 <= signal_const_114;
        195:
            signal_mux_46 <= signal_const_113;
        196:
            signal_mux_46 <= signal_const_63;
        197:
            signal_mux_46 <= signal_const_111;
        198:
            signal_mux_46 <= signal_const_77;
        199:
            signal_mux_46 <= signal_const_109;
        200:
            signal_mux_46 <= signal_const_58;
        201:
            signal_mux_46 <= signal_const_107;
        202:
            signal_mux_46 <= signal_const_106;
        203:
            signal_mux_46 <= signal_const_105;
        204:
            signal_mux_46 <= signal_const_7;
        205:
            signal_mux_46 <= signal_const_86;
        206:
            signal_mux_46 <= signal_const_102;
        207:
            signal_mux_46 <= signal_const_101;
        208:
            signal_mux_46 <= signal_const;
        209:
            signal_mux_46 <= signal_const_99;
        210:
            signal_mux_46 <= signal_const_98;
        211:
            signal_mux_46 <= signal_const_97;
        212:
            signal_mux_46 <= signal_const_96;
        213:
            signal_mux_46 <= signal_const_95;
        214:
            signal_mux_46 <= signal_const_94;
        215:
            signal_mux_46 <= signal_const_93;
        216:
            signal_mux_46 <= signal_const_92;
        217:
            signal_mux_46 <= signal_const_91;
        218:
            signal_mux_46 <= signal_const_90;
        219:
            signal_mux_46 <= signal_const_89;
        220:
            signal_mux_46 <= signal_const_71;
        221:
            signal_mux_46 <= signal_const_7;
        222:
            signal_mux_46 <= signal_const_86;
        223:
            signal_mux_46 <= signal_const_85;
        224:
            signal_mux_46 <= signal_const;
        225:
            signal_mux_46 <= signal_const_83;
        226:
            signal_mux_46 <= signal_const_82;
        227:
            signal_mux_46 <= signal_const_81;
        228:
            signal_mux_46 <= signal_const_80;
        229:
            signal_mux_46 <= signal_const_79;
        230:
            signal_mux_46 <= signal_const_78;
        231:
            signal_mux_46 <= signal_const_77;
        232:
            signal_mux_46 <= signal_const_76;
        233:
            signal_mux_46 <= signal_const_75;
        234:
            signal_mux_46 <= signal_const_74;
        235:
            signal_mux_46 <= signal_const_73;
        236:
            signal_mux_46 <= signal_const_72;
        237:
            signal_mux_46 <= signal_const_71;
        238:
            signal_mux_46 <= signal_const_7;
        239:
            signal_mux_46 <= signal_const_69;
        240:
            signal_mux_46 <= signal_const;
        241:
            signal_mux_46 <= signal_const_67;
        242:
            signal_mux_46 <= signal_const_66;
        243:
            signal_mux_46 <= signal_const_65;
        244:
            signal_mux_46 <= signal_const_64;
        245:
            signal_mux_46 <= signal_const_63;
        246:
            signal_mux_46 <= signal_const_62;
        247:
            signal_mux_46 <= signal_const_61;
        248:
            signal_mux_46 <= signal_const_60;
        249:
            signal_mux_46 <= signal_const_59;
        250:
            signal_mux_46 <= signal_const_58;
        251:
            signal_mux_46 <= signal_const_57;
        252:
            signal_mux_46 <= signal_const_56;
        253:
            signal_mux_46 <= signal_const_55;
        254:
            signal_mux_46 <= signal_const_54;
        default:
            signal_mux_46 <= signal_const_7;
        endcase
    end
    always @(posedge signal_wire_7) begin
        signal_reg_25 <= signal_mux_46;
    end
    assign signal_const_54 = 8'b00000100;
    assign signal_const_55 = 8'b00001001;
    assign signal_const_56 = 8'b00001110;
    assign signal_const_57 = 8'b00010100;
    assign signal_const_58 = 8'b00011001;
    assign signal_const_59 = 8'b00100000;
    assign signal_const_60 = 8'b00100111;
    assign signal_const_61 = 8'b00101110;
    assign signal_const_62 = 8'b00110110;
    assign signal_const_63 = 8'b01000000;
    assign signal_const_64 = 8'b01001010;
    assign signal_const_65 = 8'b01010101;
    assign signal_const_66 = 8'b01100001;
    assign signal_const_67 = 8'b01101111;
    assign signal_const_69 = 8'b11111100;
    assign signal_const_71 = 8'b00000101;
    assign signal_const_72 = 8'b00001010;
    assign signal_const_73 = 8'b00001111;
    assign signal_const_74 = 8'b00010101;
    assign signal_const_75 = 8'b00011100;
    assign signal_const_76 = 8'b00100011;
    assign signal_const_77 = 8'b00101010;
    assign signal_const_78 = 8'b00110011;
    assign signal_const_79 = 8'b00111100;
    assign signal_const_80 = 8'b01000111;
    assign signal_const_81 = 8'b01010010;
    assign signal_const_82 = 8'b01011111;
    assign signal_const_83 = 8'b01101110;
    assign signal_const_85 = 8'b11110111;
    assign signal_const_86 = 8'b11111011;
    assign signal_const_89 = 8'b00001011;
    assign signal_const_90 = 8'b00010001;
    assign signal_const_91 = 8'b00010111;
    assign signal_const_92 = 8'b00011110;
    assign signal_const_93 = 8'b00100110;
    assign signal_const_94 = 8'b00101111;
    assign signal_const_95 = 8'b00111000;
    assign signal_const_96 = 8'b01000011;
    assign signal_const_97 = 8'b01001111;
    assign signal_const_98 = 8'b01011101;
    assign signal_const_99 = 8'b01101101;
    assign signal_const_101 = 8'b11110010;
    assign signal_const_102 = 8'b11110110;
    assign signal_const_105 = 8'b00000110;
    assign signal_const_106 = 8'b00001100;
    assign signal_const_107 = 8'b00010010;
    assign signal_const_109 = 8'b00100001;
    assign signal_const_111 = 8'b00110100;
    assign signal_const_113 = 8'b01001100;
    assign signal_const_114 = 8'b01011011;
    assign signal_const_115 = 8'b01101011;
    assign signal_const_117 = 8'b11101100;
    assign signal_const_118 = 8'b11110001;
    assign signal_const_119 = 8'b11110101;
    assign signal_const_120 = 8'b11111010;
    assign signal_const_123 = 8'b00001101;
    assign signal_const_126 = 8'b00100101;
    assign signal_const_127 = 8'b00110000;
    assign signal_const_128 = 8'b00111011;
    assign signal_const_129 = 8'b01001001;
    assign signal_const_130 = 8'b01011000;
    assign signal_const_131 = 8'b01101010;
    assign signal_const_133 = 8'b11100111;
    assign signal_const_134 = 8'b11101011;
    assign signal_const_135 = 8'b11101111;
    assign signal_const_136 = 8'b11110100;
    assign signal_const_139 = 8'b00000111;
    assign signal_const_141 = 8'b00010110;
    assign signal_const_145 = 8'b01000100;
    assign signal_const_147 = 8'b01101000;
    assign signal_const_149 = 8'b11100000;
    assign signal_const_150 = 8'b11100100;
    assign signal_const_151 = 8'b11101001;
    assign signal_const_152 = 8'b11101110;
    assign signal_const_153 = 8'b11110011;
    assign signal_const_154 = 8'b11111001;
    assign signal_const_157 = 8'b00010000;
    assign signal_const_159 = 8'b00100100;
    assign signal_const_160 = 8'b00110001;
    assign signal_const_162 = 8'b01010001;
    assign signal_const_163 = 8'b01100110;
    assign signal_const_165 = 8'b11011001;
    assign signal_const_166 = 8'b11011101;
    assign signal_const_167 = 8'b11100010;
    assign signal_const_173 = 8'b00001000;
    assign signal_const_175 = 8'b00011101;
    assign signal_const_177 = 8'b00111010;
    assign signal_const_179 = 8'b01100011;
    assign signal_const_181 = 8'b11010010;
    assign signal_const_182 = 8'b11010110;
    assign signal_const_183 = 8'b11011010;
    assign signal_const_184 = 8'b11011111;
    assign signal_const_186 = 8'b11101010;
    assign signal_const_187 = 8'b11110000;
    assign signal_const_188 = 8'b11111000;
    assign signal_const_197 = 8'b11001010;
    assign signal_const_198 = 8'b11001101;
    assign signal_const_199 = 8'b11010001;
    assign signal_const_201 = 8'b11011011;
    assign signal_const_213 = 8'b11000000;
    assign signal_const_214 = 8'b11000100;
    assign signal_const_215 = 8'b11001000;
    assign signal_const_216 = 8'b11001100;
    assign signal_const_217 = 8'b11010000;
    assign signal_const_219 = 8'b11011100;
    assign signal_const_220 = 8'b11100011;
    assign signal_const_229 = 8'b10110110;
    assign signal_const_230 = 8'b10111001;
    assign signal_const_231 = 8'b10111101;
    assign signal_const_233 = 8'b11000101;
    assign signal_const_235 = 8'b11001111;
    assign signal_const_245 = 8'b10101011;
    assign signal_const_246 = 8'b10101110;
    assign signal_const_247 = 8'b10110001;
    assign signal_const_248 = 8'b10110100;
    assign signal_const_249 = 8'b10110111;
    assign signal_const_250 = 8'b10111100;
    assign signal_const_252 = 8'b11000110;
    assign signal_const_261 = 8'b10011111;
    assign signal_const_262 = 8'b10100001;
    assign signal_const_263 = 8'b10100011;
    assign signal_const_264 = 8'b10100101;
    assign signal_const_265 = 8'b10101000;
    assign signal_const_267 = 8'b10101111;
    assign signal_const_277 = 8'b10010001;
    assign signal_const_278 = 8'b10010010;
    assign signal_const_279 = 8'b10010011;
    assign signal_const_280 = 8'b10010101;
    assign signal_const_281 = 8'b10010110;
    assign signal_const_282 = 8'b10011000;
    assign signal_const_283 = 8'b10011010;
    assign signal_const_284 = 8'b10011101;
    assign signal_const_293 = 8'b10000001;
    assign signal_select_60 = signal_mux_51[33:32];
    assign signal_cat_11 = { signal_const_3,
                             signal_select_60 };
    assign signal_select_61 = signal_mux_50[33:16];
    assign signal_cat_12 = { signal_const_6,
                             signal_select_61 };
    assign signal_select_62 = signal_mux_49[33:8];
    assign signal_cat_13 = { signal_const_7,
                             signal_select_62 };
    assign signal_select_63 = signal_mux_48[33:4];
    assign signal_cat_14 = { signal_const_8,
                             signal_select_63 };
    assign signal_select_64 = signal_mux_47[33:2];
    assign signal_cat_15 = { signal_const_9,
                             signal_select_64 };
    assign signal_select_65 = depth_ask[33:1];
    assign signal_cat_16 = { signal_const_10,
                             signal_select_65 };
    assign signal_select_66 = signal_mux_92[0:0];
    assign signal_mux_47 = signal_select_66 ? signal_cat_16 : depth_ask;
    assign signal_select_67 = signal_mux_92[1:1];
    assign signal_mux_48 = signal_select_67 ? signal_cat_15 : signal_mux_47;
    assign signal_select_68 = signal_mux_92[2:2];
    assign signal_mux_49 = signal_select_68 ? signal_cat_14 : signal_mux_48;
    assign signal_select_69 = signal_mux_92[3:3];
    assign signal_mux_50 = signal_select_69 ? signal_cat_13 : signal_mux_49;
    assign signal_select_70 = signal_mux_92[4:4];
    assign signal_mux_51 = signal_select_70 ? signal_cat_12 : signal_mux_50;
    assign signal_select_71 = signal_mux_92[5:5];
    assign signal_mux_52 = signal_select_71 ? signal_cat_11 : signal_mux_51;
    assign signal_select_72 = signal_mux_52[3:0];
    always @(posedge signal_wire_7) begin
        signal_reg_26 <= signal_select_72;
    end
    assign signal_select_73 = signal_mux_57[33:32];
    assign signal_cat_17 = { signal_const_3,
                             signal_select_73 };
    assign signal_select_74 = signal_mux_56[33:16];
    assign signal_cat_18 = { signal_const_6,
                             signal_select_74 };
    assign signal_select_75 = signal_mux_55[33:8];
    assign signal_cat_19 = { signal_const_7,
                             signal_select_75 };
    assign signal_select_76 = signal_mux_54[33:4];
    assign signal_cat_20 = { signal_const_8,
                             signal_select_76 };
    assign signal_select_77 = signal_mux_53[33:2];
    assign signal_cat_21 = { signal_const_9,
                             signal_select_77 };
    assign signal_select_78 = depth_bid[33:1];
    assign signal_cat_22 = { signal_const_10,
                             signal_select_78 };
    assign signal_select_79 = signal_mux_92[0:0];
    assign signal_mux_53 = signal_select_79 ? signal_cat_22 : depth_bid;
    assign signal_select_80 = signal_mux_92[1:1];
    assign signal_mux_54 = signal_select_80 ? signal_cat_21 : signal_mux_53;
    assign signal_select_81 = signal_mux_92[2:2];
    assign signal_mux_55 = signal_select_81 ? signal_cat_20 : signal_mux_54;
    assign signal_select_82 = signal_mux_92[3:3];
    assign signal_mux_56 = signal_select_82 ? signal_cat_19 : signal_mux_55;
    assign signal_select_83 = signal_mux_92[4:4];
    assign signal_mux_57 = signal_select_83 ? signal_cat_18 : signal_mux_56;
    assign signal_sub_2 = signal_const_50 - signal_mux_91;
    assign signal_mux_58 = signal_select_137 ? signal_const_2 : signal_const_21;
    assign signal_mux_59 = signal_select_115 ? signal_const_22 : signal_const_23;
    assign signal_mux_60 = signal_or_60 ? signal_mux_58 : signal_mux_59;
    assign signal_mux_61 = signal_select_113 ? signal_const_24 : signal_const_25;
    assign signal_mux_62 = signal_select_111 ? signal_const_26 : signal_const_27;
    assign signal_mux_63 = signal_or_56 ? signal_mux_61 : signal_mux_62;
    assign signal_mux_64 = signal_or_61 ? signal_mux_60 : signal_mux_63;
    assign signal_mux_65 = signal_select_109 ? signal_const_28 : signal_const_29;
    assign signal_mux_66 = signal_select_107 ? signal_const_30 : signal_const_31;
    assign signal_mux_67 = signal_or_52 ? signal_mux_65 : signal_mux_66;
    assign signal_mux_68 = signal_select_105 ? signal_const_32 : signal_const_33;
    assign signal_mux_69 = signal_select_103 ? signal_const_34 : signal_const_35;
    assign signal_mux_70 = signal_or_49 ? signal_mux_68 : signal_mux_69;
    assign signal_mux_71 = signal_or_53 ? signal_mux_67 : signal_mux_70;
    assign signal_mux_72 = signal_or_62 ? signal_mux_64 : signal_mux_71;
    assign signal_mux_73 = signal_select_101 ? signal_const_36 : signal_const_37;
    assign signal_mux_74 = signal_select_99 ? signal_const_38 : signal_const_39;
    assign signal_mux_75 = signal_or_44 ? signal_mux_73 : signal_mux_74;
    assign signal_mux_76 = signal_select_97 ? signal_const_40 : signal_const_41;
    assign signal_mux_77 = signal_select_95 ? signal_const_42 : signal_const_43;
    assign signal_mux_78 = signal_or_41 ? signal_mux_76 : signal_mux_77;
    assign signal_mux_79 = signal_or_45 ? signal_mux_75 : signal_mux_78;
    assign signal_mux_80 = signal_select_93 ? signal_const_44 : signal_const_45;
    assign signal_mux_81 = signal_select_91 ? signal_const_46 : signal_const_47;
    assign signal_mux_82 = signal_or_37 ? signal_mux_80 : signal_mux_81;
    assign signal_mux_83 = signal_select_89 ? signal_const_18 : signal_const_49;
    assign signal_mux_84 = signal_select_87 ? signal_const_50 : signal_const_51;
    assign signal_mux_85 = signal_or_34 ? signal_mux_83 : signal_mux_84;
    assign signal_mux_86 = signal_or_38 ? signal_mux_82 : signal_mux_85;
    assign signal_mux_87 = signal_or_46 ? signal_mux_79 : signal_mux_86;
    assign signal_mux_88 = signal_or_63 ? signal_mux_72 : signal_mux_87;
    assign signal_const_357 = 6'b100001;
    assign signal_mux_89 = signal_select_85 ? signal_const_52 : signal_const_357;
    assign signal_mux_90 = signal_or_64 ? signal_mux_88 : signal_mux_89;
    assign signal_const_358 = 6'b100010;
    assign signal_select_84 = signal_or_59[0:0];
    assign signal_select_85 = signal_or_59[1:1];
    assign signal_or_32 = signal_select_85 | signal_select_84;
    assign signal_select_86 = signal_or_59[2:2];
    assign signal_select_87 = signal_or_59[3:3];
    assign signal_or_33 = signal_select_87 | signal_select_86;
    assign signal_select_88 = signal_or_59[4:4];
    assign signal_select_89 = signal_or_59[5:5];
    assign signal_or_34 = signal_select_89 | signal_select_88;
    assign signal_or_35 = signal_or_34 | signal_or_33;
    assign signal_select_90 = signal_or_59[6:6];
    assign signal_select_91 = signal_or_59[7:7];
    assign signal_or_36 = signal_select_91 | signal_select_90;
    assign signal_select_92 = signal_or_59[8:8];
    assign signal_select_93 = signal_or_59[9:9];
    assign signal_or_37 = signal_select_93 | signal_select_92;
    assign signal_or_38 = signal_or_37 | signal_or_36;
    assign signal_or_39 = signal_or_38 | signal_or_35;
    assign signal_select_94 = signal_or_59[10:10];
    assign signal_select_95 = signal_or_59[11:11];
    assign signal_or_40 = signal_select_95 | signal_select_94;
    assign signal_select_96 = signal_or_59[12:12];
    assign signal_select_97 = signal_or_59[13:13];
    assign signal_or_41 = signal_select_97 | signal_select_96;
    assign signal_or_42 = signal_or_41 | signal_or_40;
    assign signal_select_98 = signal_or_59[14:14];
    assign signal_select_99 = signal_or_59[15:15];
    assign signal_or_43 = signal_select_99 | signal_select_98;
    assign signal_select_100 = signal_or_59[16:16];
    assign signal_select_101 = signal_or_59[17:17];
    assign signal_or_44 = signal_select_101 | signal_select_100;
    assign signal_or_45 = signal_or_44 | signal_or_43;
    assign signal_or_46 = signal_or_45 | signal_or_42;
    assign signal_or_47 = signal_or_46 | signal_or_39;
    assign signal_select_102 = signal_or_59[18:18];
    assign signal_select_103 = signal_or_59[19:19];
    assign signal_or_48 = signal_select_103 | signal_select_102;
    assign signal_select_104 = signal_or_59[20:20];
    assign signal_select_105 = signal_or_59[21:21];
    assign signal_or_49 = signal_select_105 | signal_select_104;
    assign signal_or_50 = signal_or_49 | signal_or_48;
    assign signal_select_106 = signal_or_59[22:22];
    assign signal_select_107 = signal_or_59[23:23];
    assign signal_or_51 = signal_select_107 | signal_select_106;
    assign signal_select_108 = signal_or_59[24:24];
    assign signal_select_109 = signal_or_59[25:25];
    assign signal_or_52 = signal_select_109 | signal_select_108;
    assign signal_or_53 = signal_or_52 | signal_or_51;
    assign signal_or_54 = signal_or_53 | signal_or_50;
    assign signal_select_110 = signal_or_59[26:26];
    assign signal_select_111 = signal_or_59[27:27];
    assign signal_or_55 = signal_select_111 | signal_select_110;
    assign signal_select_112 = signal_or_59[28:28];
    assign signal_select_113 = signal_or_59[29:29];
    assign signal_or_56 = signal_select_113 | signal_select_112;
    assign signal_or_57 = signal_or_56 | signal_or_55;
    assign signal_select_114 = signal_or_59[30:30];
    assign signal_select_115 = signal_or_59[31:31];
    assign signal_or_58 = signal_select_115 | signal_select_114;
    assign signal_select_116 = signal_or_59[32:32];
    assign signal_select_117 = signal_select_119[127:64];
    assign signal_select_118 = signal_select_117[63:32];
    assign signal_cat_23 = { signal_const_9,
                             signal_select_118 };
    assign signal_select_119 = signal_select_122[191:64];
    assign signal_select_120 = signal_select_119[63:0];
    assign signal_select_121 = signal_select_120[63:32];
    assign signal_cat_24 = { signal_const_9,
                             signal_select_121 };
    assign signal_select_122 = signal_wire_3[255:64];
    assign signal_select_123 = signal_select_122[63:0];
    assign signal_select_124 = signal_select_123[63:32];
    assign signal_cat_25 = { signal_const_9,
                             signal_select_124 };
    assign signal_wire_3 = book$asks;
    assign signal_select_125 = signal_wire_3[63:0];
    assign signal_select_126 = signal_select_125[63:32];
    assign signal_cat_26 = { signal_const_9,
                             signal_select_126 };
    assign signal_add = signal_cat_26 + signal_cat_25;
    assign signal_add_1 = signal_add + signal_cat_24;
    assign signal_add_2 = signal_add_1 + signal_cat_23;
    always @(posedge signal_wire_7) begin
        depth_ask <= signal_add_2;
    end
    assign signal_select_127 = signal_select_129[127:64];
    assign signal_select_128 = signal_select_127[63:32];
    assign signal_cat_27 = { signal_const_9,
                             signal_select_128 };
    assign signal_select_129 = signal_select_132[191:64];
    assign signal_select_130 = signal_select_129[63:0];
    assign signal_select_131 = signal_select_130[63:32];
    assign signal_cat_28 = { signal_const_9,
                             signal_select_131 };
    assign signal_select_132 = signal_wire_4[255:64];
    assign signal_select_133 = signal_select_132[63:0];
    assign signal_select_134 = signal_select_133[63:32];
    assign signal_cat_29 = { signal_const_9,
                             signal_select_134 };
    assign signal_wire_4 = book$bids;
    assign signal_select_135 = signal_wire_4[63:0];
    assign signal_select_136 = signal_select_135[63:32];
    assign signal_cat_30 = { signal_const_9,
                             signal_select_136 };
    assign signal_add_3 = signal_cat_30 + signal_cat_29;
    assign signal_add_4 = signal_add_3 + signal_cat_28;
    assign signal_add_5 = signal_add_4 + signal_cat_27;
    always @(posedge signal_wire_7) begin
        depth_bid <= signal_add_5;
    end
    assign signal_or_59 = depth_bid | depth_ask;
    assign signal_select_137 = signal_or_59[33:33];
    assign signal_or_60 = signal_select_137 | signal_select_116;
    assign signal_or_61 = signal_or_60 | signal_or_58;
    assign signal_or_62 = signal_or_61 | signal_or_57;
    assign signal_or_63 = signal_or_62 | signal_or_54;
    assign signal_or_64 = signal_or_63 | signal_or_47;
    assign signal_or_65 = signal_or_64 | signal_or_32;
    assign signal_mux_91 = signal_or_65 ? signal_mux_90 : signal_const_358;
    assign signal_lt_2 = signal_mux_91 < signal_const_50;
    assign signal_not_3 = ~ signal_lt_2;
    assign signal_mux_92 = signal_not_3 ? signal_const_2 : signal_sub_2;
    assign signal_select_138 = signal_mux_92[5:5];
    assign signal_mux_93 = signal_select_138 ? signal_cat_17 : signal_mux_57;
    assign signal_select_139 = signal_mux_93[3:0];
    always @(posedge signal_wire_7) begin
        signal_reg_27 <= signal_select_139;
    end
    assign depth_index = { signal_reg_27,
                           signal_reg_26 };
    always @* begin
        case (depth_index)
        0:
            signal_mux_94 <= signal_const_7;
        1:
            signal_mux_94 <= signal_const_293;
        2:
            signal_mux_94 <= signal_const_293;
        3:
            signal_mux_94 <= signal_const_293;
        4:
            signal_mux_94 <= signal_const_293;
        5:
            signal_mux_94 <= signal_const_293;
        6:
            signal_mux_94 <= signal_const_293;
        7:
            signal_mux_94 <= signal_const_293;
        8:
            signal_mux_94 <= signal_const_293;
        9:
            signal_mux_94 <= signal_const_293;
        10:
            signal_mux_94 <= signal_const_293;
        11:
            signal_mux_94 <= signal_const_293;
        12:
            signal_mux_94 <= signal_const_293;
        13:
            signal_mux_94 <= signal_const_293;
        14:
            signal_mux_94 <= signal_const_293;
        15:
            signal_mux_94 <= signal_const_293;
        16:
            signal_mux_94 <= signal_const;
        17:
            signal_mux_94 <= signal_const_7;
        18:
            signal_mux_94 <= signal_const_182;
        19:
            signal_mux_94 <= signal_const_213;
        20:
            signal_mux_94 <= signal_const_248;
        21:
            signal_mux_94 <= signal_const_245;
        22:
            signal_mux_94 <= signal_const_264;
        23:
            signal_mux_94 <= signal_const_262;
        24:
            signal_mux_94 <= signal_const_284;
        25:
            signal_mux_94 <= signal_const_283;
        26:
            signal_mux_94 <= signal_const_282;
        27:
            signal_mux_94 <= signal_const_281;
        28:
            signal_mux_94 <= signal_const_280;
        29:
            signal_mux_94 <= signal_const_279;
        30:
            signal_mux_94 <= signal_const_278;
        31:
            signal_mux_94 <= signal_const_277;
        32:
            signal_mux_94 <= signal_const;
        33:
            signal_mux_94 <= signal_const_77;
        34:
            signal_mux_94 <= signal_const_7;
        35:
            signal_mux_94 <= signal_const_133;
        36:
            signal_mux_94 <= signal_const_182;
        37:
            signal_mux_94 <= signal_const_197;
        38:
            signal_mux_94 <= signal_const_213;
        39:
            signal_mux_94 <= signal_const_230;
        40:
            signal_mux_94 <= signal_const_248;
        41:
            signal_mux_94 <= signal_const_267;
        42:
            signal_mux_94 <= signal_const_245;
        43:
            signal_mux_94 <= signal_const_265;
        44:
            signal_mux_94 <= signal_const_264;
        45:
            signal_mux_94 <= signal_const_263;
        46:
            signal_mux_94 <= signal_const_262;
        47:
            signal_mux_94 <= signal_const_261;
        48:
            signal_mux_94 <= signal_const;
        49:
            signal_mux_94 <= signal_const_63;
        50:
            signal_mux_94 <= signal_const_58;
        51:
            signal_mux_94 <= signal_const_7;
        52:
            signal_mux_94 <= signal_const_152;
        53:
            signal_mux_94 <= signal_const_149;
        54:
            signal_mux_94 <= signal_const_182;
        55:
            signal_mux_94 <= signal_const_198;
        56:
            signal_mux_94 <= signal_const_252;
        57:
            signal_mux_94 <= signal_const_213;
        58:
            signal_mux_94 <= signal_const_250;
        59:
            signal_mux_94 <= signal_const_249;
        60:
            signal_mux_94 <= signal_const_248;
        61:
            signal_mux_94 <= signal_const_247;
        62:
            signal_mux_94 <= signal_const_246;
        63:
            signal_mux_94 <= signal_const_245;
        64:
            signal_mux_94 <= signal_const;
        65:
            signal_mux_94 <= signal_const_113;
        66:
            signal_mux_94 <= signal_const_77;
        67:
            signal_mux_94 <= signal_const_107;
        68:
            signal_mux_94 <= signal_const_7;
        69:
            signal_mux_94 <= signal_const_101;
        70:
            signal_mux_94 <= signal_const_133;
        71:
            signal_mux_94 <= signal_const_166;
        72:
            signal_mux_94 <= signal_const_182;
        73:
            signal_mux_94 <= signal_const_235;
        74:
            signal_mux_94 <= signal_const_197;
        75:
            signal_mux_94 <= signal_const_233;
        76:
            signal_mux_94 <= signal_const_213;
        77:
            signal_mux_94 <= signal_const_231;
        78:
            signal_mux_94 <= signal_const_230;
        79:
            signal_mux_94 <= signal_const_229;
        80:
            signal_mux_94 <= signal_const;
        81:
            signal_mux_94 <= signal_const_65;
        82:
            signal_mux_94 <= signal_const_62;
        83:
            signal_mux_94 <= signal_const_59;
        84:
            signal_mux_94 <= signal_const_56;
        85:
            signal_mux_94 <= signal_const_7;
        86:
            signal_mux_94 <= signal_const_136;
        87:
            signal_mux_94 <= signal_const_134;
        88:
            signal_mux_94 <= signal_const_220;
        89:
            signal_mux_94 <= signal_const_219;
        90:
            signal_mux_94 <= signal_const_182;
        91:
            signal_mux_94 <= signal_const_217;
        92:
            signal_mux_94 <= signal_const_216;
        93:
            signal_mux_94 <= signal_const_215;
        94:
            signal_mux_94 <= signal_const_214;
        95:
            signal_mux_94 <= signal_const_213;
        96:
            signal_mux_94 <= signal_const;
        97:
            signal_mux_94 <= signal_const_114;
        98:
            signal_mux_94 <= signal_const_63;
        99:
            signal_mux_94 <= signal_const_77;
        100:
            signal_mux_94 <= signal_const_58;
        101:
            signal_mux_94 <= signal_const_106;
        102:
            signal_mux_94 <= signal_const_7;
        103:
            signal_mux_94 <= signal_const_102;
        104:
            signal_mux_94 <= signal_const_152;
        105:
            signal_mux_94 <= signal_const_133;
        106:
            signal_mux_94 <= signal_const_149;
        107:
            signal_mux_94 <= signal_const_201;
        108:
            signal_mux_94 <= signal_const_182;
        109:
            signal_mux_94 <= signal_const_199;
        110:
            signal_mux_94 <= signal_const_198;
        111:
            signal_mux_94 <= signal_const_197;
        112:
            signal_mux_94 <= signal_const;
        113:
            signal_mux_94 <= signal_const_82;
        114:
            signal_mux_94 <= signal_const_80;
        115:
            signal_mux_94 <= signal_const_78;
        116:
            signal_mux_94 <= signal_const_76;
        117:
            signal_mux_94 <= signal_const_74;
        118:
            signal_mux_94 <= signal_const_72;
        119:
            signal_mux_94 <= signal_const_7;
        120:
            signal_mux_94 <= signal_const_188;
        121:
            signal_mux_94 <= signal_const_187;
        122:
            signal_mux_94 <= signal_const_186;
        123:
            signal_mux_94 <= signal_const_150;
        124:
            signal_mux_94 <= signal_const_184;
        125:
            signal_mux_94 <= signal_const_183;
        126:
            signal_mux_94 <= signal_const_182;
        127:
            signal_mux_94 <= signal_const_181;
        128:
            signal_mux_94 <= signal_const;
        129:
            signal_mux_94 <= signal_const_179;
        130:
            signal_mux_94 <= signal_const_113;
        131:
            signal_mux_94 <= signal_const_177;
        132:
            signal_mux_94 <= signal_const_77;
        133:
            signal_mux_94 <= signal_const_175;
        134:
            signal_mux_94 <= signal_const_107;
        135:
            signal_mux_94 <= signal_const_173;
        136:
            signal_mux_94 <= signal_const_7;
        137:
            signal_mux_94 <= signal_const_154;
        138:
            signal_mux_94 <= signal_const_101;
        139:
            signal_mux_94 <= signal_const_117;
        140:
            signal_mux_94 <= signal_const_133;
        141:
            signal_mux_94 <= signal_const_167;
        142:
            signal_mux_94 <= signal_const_166;
        143:
            signal_mux_94 <= signal_const_165;
        144:
            signal_mux_94 <= signal_const;
        145:
            signal_mux_94 <= signal_const_163;
        146:
            signal_mux_94 <= signal_const_162;
        147:
            signal_mux_94 <= signal_const_63;
        148:
            signal_mux_94 <= signal_const_160;
        149:
            signal_mux_94 <= signal_const_159;
        150:
            signal_mux_94 <= signal_const_58;
        151:
            signal_mux_94 <= signal_const_157;
        152:
            signal_mux_94 <= signal_const_139;
        153:
            signal_mux_94 <= signal_const_7;
        154:
            signal_mux_94 <= signal_const_154;
        155:
            signal_mux_94 <= signal_const_153;
        156:
            signal_mux_94 <= signal_const_152;
        157:
            signal_mux_94 <= signal_const_151;
        158:
            signal_mux_94 <= signal_const_150;
        159:
            signal_mux_94 <= signal_const_149;
        160:
            signal_mux_94 <= signal_const;
        161:
            signal_mux_94 <= signal_const_147;
        162:
            signal_mux_94 <= signal_const_65;
        163:
            signal_mux_94 <= signal_const_145;
        164:
            signal_mux_94 <= signal_const_62;
        165:
            signal_mux_94 <= signal_const_77;
        166:
            signal_mux_94 <= signal_const_59;
        167:
            signal_mux_94 <= signal_const_141;
        168:
            signal_mux_94 <= signal_const_56;
        169:
            signal_mux_94 <= signal_const_139;
        170:
            signal_mux_94 <= signal_const_7;
        171:
            signal_mux_94 <= signal_const_120;
        172:
            signal_mux_94 <= signal_const_136;
        173:
            signal_mux_94 <= signal_const_135;
        174:
            signal_mux_94 <= signal_const_134;
        175:
            signal_mux_94 <= signal_const_133;
        176:
            signal_mux_94 <= signal_const;
        177:
            signal_mux_94 <= signal_const_131;
        178:
            signal_mux_94 <= signal_const_130;
        179:
            signal_mux_94 <= signal_const_129;
        180:
            signal_mux_94 <= signal_const_128;
        181:
            signal_mux_94 <= signal_const_127;
        182:
            signal_mux_94 <= signal_const_126;
        183:
            signal_mux_94 <= signal_const_75;
        184:
            signal_mux_94 <= signal_const_57;
        185:
            signal_mux_94 <= signal_const_123;
        186:
            signal_mux_94 <= signal_const_105;
        187:
            signal_mux_94 <= signal_const_7;
        188:
            signal_mux_94 <= signal_const_120;
        189:
            signal_mux_94 <= signal_const_119;
        190:
            signal_mux_94 <= signal_const_118;
        191:
            signal_mux_94 <= signal_const_117;
        192:
            signal_mux_94 <= signal_const;
        193:
            signal_mux_94 <= signal_const_115;
        194:
            signal_mux_94 <= signal_const_114;
        195:
            signal_mux_94 <= signal_const_113;
        196:
            signal_mux_94 <= signal_const_63;
        197:
            signal_mux_94 <= signal_const_111;
        198:
            signal_mux_94 <= signal_const_77;
        199:
            signal_mux_94 <= signal_const_109;
        200:
            signal_mux_94 <= signal_const_58;
        201:
            signal_mux_94 <= signal_const_107;
        202:
            signal_mux_94 <= signal_const_106;
        203:
            signal_mux_94 <= signal_const_105;
        204:
            signal_mux_94 <= signal_const_7;
        205:
            signal_mux_94 <= signal_const_86;
        206:
            signal_mux_94 <= signal_const_102;
        207:
            signal_mux_94 <= signal_const_101;
        208:
            signal_mux_94 <= signal_const;
        209:
            signal_mux_94 <= signal_const_99;
        210:
            signal_mux_94 <= signal_const_98;
        211:
            signal_mux_94 <= signal_const_97;
        212:
            signal_mux_94 <= signal_const_96;
        213:
            signal_mux_94 <= signal_const_95;
        214:
            signal_mux_94 <= signal_const_94;
        215:
            signal_mux_94 <= signal_const_93;
        216:
            signal_mux_94 <= signal_const_92;
        217:
            signal_mux_94 <= signal_const_91;
        218:
            signal_mux_94 <= signal_const_90;
        219:
            signal_mux_94 <= signal_const_89;
        220:
            signal_mux_94 <= signal_const_71;
        221:
            signal_mux_94 <= signal_const_7;
        222:
            signal_mux_94 <= signal_const_86;
        223:
            signal_mux_94 <= signal_const_85;
        224:
            signal_mux_94 <= signal_const;
        225:
            signal_mux_94 <= signal_const_83;
        226:
            signal_mux_94 <= signal_const_82;
        227:
            signal_mux_94 <= signal_const_81;
        228:
            signal_mux_94 <= signal_const_80;
        229:
            signal_mux_94 <= signal_const_79;
        230:
            signal_mux_94 <= signal_const_78;
        231:
            signal_mux_94 <= signal_const_77;
        232:
            signal_mux_94 <= signal_const_76;
        233:
            signal_mux_94 <= signal_const_75;
        234:
            signal_mux_94 <= signal_const_74;
        235:
            signal_mux_94 <= signal_const_73;
        236:
            signal_mux_94 <= signal_const_72;
        237:
            signal_mux_94 <= signal_const_71;
        238:
            signal_mux_94 <= signal_const_7;
        239:
            signal_mux_94 <= signal_const_69;
        240:
            signal_mux_94 <= signal_const;
        241:
            signal_mux_94 <= signal_const_67;
        242:
            signal_mux_94 <= signal_const_66;
        243:
            signal_mux_94 <= signal_const_65;
        244:
            signal_mux_94 <= signal_const_64;
        245:
            signal_mux_94 <= signal_const_63;
        246:
            signal_mux_94 <= signal_const_62;
        247:
            signal_mux_94 <= signal_const_61;
        248:
            signal_mux_94 <= signal_const_60;
        249:
            signal_mux_94 <= signal_const_59;
        250:
            signal_mux_94 <= signal_const_58;
        251:
            signal_mux_94 <= signal_const_57;
        252:
            signal_mux_94 <= signal_const_56;
        253:
            signal_mux_94 <= signal_const_55;
        254:
            signal_mux_94 <= signal_const_54;
        default:
            signal_mux_94 <= signal_const_7;
        endcase
    end
    always @(posedge signal_wire_7) begin
        signal_reg_28 <= signal_mux_94;
    end
    assign signal_select_140 = signal_cat_35[7:0];
    assign signal_const_369 = 20'b01111111111110000001;
    assign signal_select_141 = signal_cat_35[18:0];
    assign signal_select_142 = signal_cat_35[19:19];
    assign signal_not_4 = ~ signal_select_142;
    assign signal_cat_31 = { signal_not_4,
                             signal_select_141 };
    assign signal_lt_3 = signal_cat_31 < signal_const_369;
    assign signal_mux_95 = signal_lt_3 ? signal_const_293 : signal_select_140;
    assign signal_select_143 = signal_cat_35[18:0];
    assign signal_select_144 = signal_wire_5[19:5];
    assign signal_wire_5 = book$flow;
    assign signal_select_145 = signal_wire_5[19:19];
    assign signal_cat_32 = { signal_select_145,
                             signal_select_145 };
    assign signal_cat_33 = { signal_cat_32,
                             signal_cat_32 };
    assign signal_cat_34 = { signal_cat_33,
                             signal_select_145 };
    assign signal_cat_35 = { signal_cat_34,
                             signal_select_144 };
    assign signal_select_146 = signal_cat_35[19:19];
    assign signal_not_5 = ~ signal_select_146;
    assign signal_cat_36 = { signal_not_5,
                             signal_select_143 };
    assign signal_const_370 = 20'b10000000000001111111;
    assign signal_lt_4 = signal_const_370 < signal_cat_36;
    assign signal_mux_96 = signal_lt_4 ? signal_const : signal_mux_95;
    always @(posedge signal_wire_7) begin
        flow <= signal_mux_96;
    end
    always @(posedge signal_wire_7) begin
        signal_reg_29 <= flow;
    end
    always @(posedge signal_wire_7) begin
        signal_reg_30 <= signal_reg_29;
    end
    assign signal_cat_37 = { signal_reg_30,
                             signal_reg_28,
                             signal_reg_25,
                             signal_reg_22 };
    assign signal_wire_6 = clear;
    assign signal_wire_7 = clock;
    assign signal_wire_8 = book$valid;
    always @(posedge signal_wire_7) begin
        if (signal_wire_6)
            signal_reg_31 <= signal_const_10;
        else
            signal_reg_31 <= signal_wire_8;
    end
    always @(posedge signal_wire_7) begin
        if (signal_wire_6)
            signal_reg_32 <= signal_const_10;
        else
            signal_reg_32 <= signal_reg_31;
    end
    always @(posedge signal_wire_7) begin
        if (signal_wire_6)
            signal_reg_33 <= signal_const_10;
        else
            signal_reg_33 <= signal_reg_32;
    end
    assign features$valid = signal_reg_33;
    assign features$value = signal_cat_37;
    assign context$symbol = signal_reg_20;
    assign context$bid = signal_reg_17;
    assign context$has_bid = signal_reg_14;
    assign context$ask = signal_reg_11;
    assign context$has_ask = signal_reg_8;
    assign context$t_first = signal_reg_5;
    assign context$t_last = signal_reg_2;

endmodule
