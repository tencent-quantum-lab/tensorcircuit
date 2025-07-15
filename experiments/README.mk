# 在 Tensorcircuit 中使用参数化波形完成 Rabi 实验

## 下载安装
```
git clone https://github.com/ruan-prog/tensorcircuit.git
cd tensorcircuit
git checkout rabi_test

cd experiments
pip install -r requirements.txt
```

## 代码实现

在 rabi.py 文件中修改代码
```
vim rabi.py
```

#### 1. 配置量子云服务

```python
set_token("your_api_token")  # 需替换为真实的量子云服务令牌
```

#### 2. 自定义参数化波形


```python
def gen_parametric_waveform_circuit(t):
    """
    参数:
        t : 脉冲持续时间（dt）
    
    返回:
        Circuit: 包含自定义波形的量子电路
    """
    qc = Circuit(1)

    param0 = Param("a")

    builder = qc.calibrate("rabi_test", [param0])
    builder.new_frame("drive_frame", param0)
    builder.play("drive_frame", waveforms.CosineDrag(t, 0.2, 0.0, 0.0))

    builder.build()
    qc.add_calibration('rabi_test', ['q[0]']) 
    
    tqasm_code = qc.to_tqasm()

    print(tqasm_code)
    return qc
```


#### 3. 在量子设备中执行电路

通过以下输出获取可用的设备名称
```python
ds = list_devices()
```

在其中一台设备上执行电路
```python
def run_circuit(qc):
    """   
    参数:
        qc (Circuit): 待执行量子电路
    
    返回:
        rf(Dict): 测量结果统计
    """
    device_name = "tianji_m2" 
    d = get_device(device_name)
    t = submit_task(
    circuit=qc,
    shots=shots_const,
    device=d,
    enable_qos_gate_decomposition=False,
    enable_qos_qubit_mapping=False,
    )
    rf = t.results()
    return rf
```

#### 4. Rabi 参数扫描

定义扫描周期，遍历不同的脉冲长度
```python
def exp_rabi():
    result_lst = []
    for t in range(1, 400, 2):
        qc = gen_parametric_waveform_circuit(t)
        result = run_circuit(qc)
        result['duration'] = t
        result_lst.append(result)
    return result_lst
```

#### 5. 绘制 Rabi 结果图

绘制 0/1 态的结果分布
```python
def draw_rabi(result_lst):
    data = {
        'duration': [],
        '0': [],
        '1': []
    }
    
    for result in result_lst:
        data['0'].append(int(result['0']) / shots_const)
        data['1'].append(int(result['1']) / shots_const)
        data['duration'].append(result['duration'])

    plt.figure(figsize=(10,6))
    plt.plot(data['duration'], data['0'], 'b-o', label='State |0>')
    plt.plot(data['duration'], data['1'], 'r--s', label='State |1>')

    plt.title('Rabi Oscillation Experiment')
    plt.xlabel('Duration (dt)')
    plt.ylabel('Probability')
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()

    plt.savefig('rabi.png', dpi=300)
    plt.show()
```
