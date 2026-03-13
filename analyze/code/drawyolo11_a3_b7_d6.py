from pathlib import Path

mermaid_code = r"""
flowchart LR

    %% 样式
    classDef default fill:white,stroke:black,stroke-width:1px,color:black
    classDef enhance fill:white,stroke:black,stroke-width:2px,color:black,font-weight:bold
    classDef group fill:#f6f6e8,stroke:#999,stroke-width:1px,color:black

    %% =========================
    %% Backbone
    %% =========================
    subgraph BB["Backbone（骨干网络）"]
        direction TB
        B0[Conv]
        B1[C3k2]
        A3[["a3: SPD 下采样"]]
        P3B["C3k2 / P3"]
        P4B["C3k2 / P4"]
        P5B["C3k2 / P5"]
        SPPF[SPPF]

        B0 --> B1 --> A3 --> P3B --> P4B --> P5B --> SPPF
    end
    class BB group

    %% =========================
    %% Neck：按参考图做成两列
    %% 左列：top-down
    %% 右列：bottom-up
    %% =========================
    subgraph NK["Neck（特征融合）"]
        direction LR

        subgraph NKL[" "]
            direction TB
            U1[Upsample]
            C1[Concat]
            N1[C3k2]

            B7[["b7: CARAFE 上采样"]]
            C2[Concat]
            N2["C3k2 / P3 neck"]

            U1 --> C1 --> N1 --> B7 --> C2 --> N2
        end

        subgraph NKR[" "]
            direction TB
            D1[Conv S2]
            C3[Concat]
            N3["C3k2 / P4 neck"]

            D2[Conv S2]
            C4[Concat]
            N4["C3k2 / P5 neck"]

            D1 --> C3 --> N3 --> D2 --> C4 --> N4
        end
    end
    class NK group

    %% =========================
    %% Head
    %% d6 只在 P3 head
    %% =========================
    subgraph HD["Head（检测头）"]
        direction TB
        D6[["d6: P3 分类分数校准"]]
        H3[P3 Head]
        H4[P4 Head]
        H5[P5 Head]

        D6 --> H3
        H4
        H5
    end
    class HD group

    %% =========================
    %% Backbone -> Neck
    %% =========================
    SPPF --> U1
    P4B --> C1
    P3B --> C2

    %% =========================
    %% Neck 内部回流
    %% =========================
    N1 --> D1
    P4B --> C3
    N3 --> D2
    P5B --> C4

    %% =========================
    %% Neck -> Head
    %% =========================
    N2 --> D6
    N3 --> H4
    N4 --> H5

    %% 样式
    class B0,B1,P3B,P4B,P5B,SPPF,U1,C1,N1,C2,N2,D1,C3,N3,D2,C4,N4,H3,H4,H5 default
    class A3,B7,D6 enhance
"""

out_path = Path("/home/ubuntu/hpproject/yolo/analyze/result/report_2603121730/yolo11_a3_b7_d6_layout_fixed.mmd")
out_path.write_text(mermaid_code, encoding="utf-8")
print(f"已写入: {out_path.resolve()}")