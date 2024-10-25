from tkinter import filedialog
import tkinter as tk
import numpy as np
import pandas as pd


def data_pro():  # 批量归一化
    from scipy.interpolate import interp1d
    from tkinter import simpledialog
    file_CSV1 = filedialog.askopenfilename(title='请选择一个标准时间轴csv文件', filetypes=[("Csv files", "*.csv")])
    if file_CSV1:
        df = pd.read_csv(file_CSV1, header=None)
        df = df.drop(df.index[:2])
        df = df.reset_index(drop=True)
        L_t = len(df[0])
        L_V1 = len(df[2])
        L = min(L_t, L_V1)

        t = df[0][:L]
        t = t.astype(float)
        t = t - 7.2e-9  # 待确定, 时间轴偏置量
        file_CSV2 = filedialog.askopenfilename(title='请选择一个扫描csv文件', filetypes=[("Csv files", "*.csv")])
        if file_CSV2:
            data = pd.read_csv(file_CSV2, header=None)
            row = len(data)
            test_data = pd.DataFrame()
            initial_value = "5 1500 300"
            inputs = simpledialog.askstring("归一化后参数选取时间轴", "请输入时间序列起始点,结束点,频率(用空格隔开):",
                                            initialvalue=initial_value)
            values = inputs.split()
            values = list(map(int, values))
            user_stp = int(values[0])
            user_edp = int(values[1])
            user_fn = int(values[2])
            for i in range(row):
                V1 = data.iloc[i, :L]
                V1 = V1.astype(float)
                V1 = V1 - V1[0]

                val, inx = np.abs(V1).max(), np.abs(V1).argmax()
                # t = t + 12e-9 - t[inx]
                val, inx = np.abs(t - -1e-10).min(), np.abs(t - -1e-10).argmin()
                base_V1 = np.mean(V1[:inx])
                V1 = (V1 - base_V1)
                val, inx = np.abs(V1).max(), np.abs(V1).argmax()
                V1 = V1 / V1[inx]  # 在此处结束了normalise。
                T = np.column_stack((t * 1e9, V1))

                firstpoint = np.where(T[:, 0] > 1)[0][0]
                # endpoint = np.where(T[:, 0] == 2400)[0][0]
                V11 = V1[firstpoint:]
                t1 = t[firstpoint:]
                x = np.logspace(np.log10(user_stp), np.log10(user_edp), user_fn)
                interp_func = interp1d(t1 * 1e9, V11, kind='linear', fill_value='extrapolate')
                y = interp_func(x)
                merged = pd.Series(y)
                test_data = test_data.append(merged, ignore_index=True)
            save_path = filedialog.asksaveasfilename(defaultextension='.csv', filetypes=[("CSV Files", "*.csv")])

            # 选择要保存的文件路径
            if save_path:
                # 将merged_df保存为CSV文件
                test_data.to_csv(save_path, index=False, header=None)
                tk.messagebox.showinfo('提示', '测试数据保存成功')
            else:
                tk.messagebox.showinfo('提示', '未选择保存路径！')
        else:
            tk.messagebox.showinfo('错误', '未检测到添加的实验数据csv文件')
    else:
        tk.messagebox.showinfo('错误', '未检测到添加的原始csv文件')


if __name__ == "__main__":
    data_pro()
