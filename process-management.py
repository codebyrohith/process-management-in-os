from collections import deque
from queue import Queue
import numpy as np
import psutil
import tkinter as tk
from tkinter import ttk
import threading
import time
import matplotlib.pyplot as plt

SHORT_BURST_THRESHOLD = 0.5
LONG_BURST_THRESHOLD = 1.0

class Scheduler:
    def __init__(self, processes):
        self.processes = deque(processes)

    def round_robin(self, time_quantum):
        print('Scheduling using Round Robin: ')
        ready_queue = Queue()
        for process in self.processes:
            ready_queue.put(process)

        while not ready_queue.empty():
            process = ready_queue.get()
            print(f"Running process {process['pid']} for time quantum {time_quantum}")
            # Simulate process execution
            if process['cpu_burst'] > time_quantum:
                process['cpu_burst'] -= time_quantum
                ready_queue.put(process)
            else:
                process['cpu_burst'] = 0
                print(f"Process {process['pid']} completed")


    def first_come_first_serve(self):
        print('Scheduling using First Come First Serve: ')
        sorted_processes = sorted(self.processes, key=lambda x: x['create_time'])
        for process in sorted_processes:
            print(f"Running process {process['pid']}")
            print(f"Process {process['pid']} completed")


    def shortest_job_first(self):
        print('Scheduling using Shortest Job First: ')
        sorted_processes = sorted(self.processes, key=lambda x: x['cpu_percent'])
        for process in sorted_processes:
            print(f"Running process {process['pid']}")
            print(f"Process {process['pid']} completed")


    def calculate_fairness(self):
        if not self.processes:
            return 0

        cpu_percentages = [process['cpu_percent'] for process in self.processes]
        n = len(cpu_percentages)

        sorted_percentages = np.sort(cpu_percentages)
        sum_percentages = np.sum(sorted_percentages)

        if sum_percentages == 0:
            return 0

        G = 2 * np.sum((np.arange(1, n + 1) * sorted_percentages)) / (n * sum_percentages) - (n + 1) / n

        return G

    def calculate_throughput(self, elapsed_time):
        num_processes = len(self.processes)
        throughput = num_processes / elapsed_time if elapsed_time > 0 else 0
        return throughput

    def calculate_response_time(self, elapsed_time):
        num_processes = len(self.processes)
        response_time = elapsed_time / num_processes if num_processes > 0 else 0
        return response_time

def categorize_processes():
    user_processes = []
    system_processes = []
    for proc in psutil.process_iter(['pid', 'name', 'username', 'ppid']):
        try:
            process_info = proc.info

            if is_system_process(process_info):
                system_processes.append(process_info)
            else:
                user_processes.append(process_info)
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass
    return user_processes, system_processes

def is_system_process(process_info):

    if process_info['username'] is None:
        return True
    elif process_info['pid'] == 0 or process_info['username'] == 'root':
        return True
    else:
        return False

def classify_bound(process_info):
    if process_info['cpu_percent'] > 70:
        return 'CPU'
    elif process_info['read_bytes'] > 1000000 or process_info['write_bytes'] > 1000000:
        return 'I/O'
    else:
        return 'Unknown'

def classify_burst_duration(cpu_times):
    burst_duration = cpu_times[1] - cpu_times[0]
    if burst_duration <= SHORT_BURST_THRESHOLD:
        return 'Short'
    elif burst_duration > LONG_BURST_THRESHOLD:
        return 'Long'
    else:
        return 'Unknown'

def gather_process_info(proc):
    try:
        info = proc.info
        cpu_percent = proc.cpu_percent(interval=0.1)
        io_counters = proc.io_counters()
        cpu_times = proc.cpu_times()
        memory_percent = proc.memory_percent()
        create_time = proc.create_time()

        info['cpu_percent'] = cpu_percent
        info['memory_percent'] = memory_percent
        info['read_bytes'] = io_counters.read_bytes if io_counters else 0
        info['write_bytes'] = io_counters.write_bytes if io_counters else 0
        info['burst_duration'] = classify_burst_duration(cpu_times)
        info['bound'] = classify_bound(info)
        info['cpu_burst'] = np.random.randint(1, 10)
        info['create_time'] = create_time

        return info
    except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
        return None
def gather_utilization_info():
    user_process_info = []
    system_process_info = []

    for proc in psutil.process_iter(['pid', 'name', 'username', 'cpu_percent', 'memory_percent']):
        process_info = gather_process_info(proc)
        if process_info:
            if is_system_process(process_info):
                system_process_info.append(process_info)
            else:
                user_process_info.append(process_info)

    return user_process_info, system_process_info


class ProcessMonitor:
    def __init__(self, root):
        self.root = root
        self.root.title("Process Monitor")
        self.root.geometry("800x600")

        self.process_tree = ttk.Treeview(self.root, columns=('pid', 'name', 'cpu_percent', 'memory_percent',
                                                             'read_bytes', 'write_bytes', 'bound', 'burst_duration'))
        self.process_tree.heading('pid', text='PID')
        self.process_tree.heading('name', text='Name')
        self.process_tree.heading('cpu_percent', text='CPU %')
        self.process_tree.heading('memory_percent', text='Memory %')
        self.process_tree.heading('read_bytes', text='Read Bytes')
        self.process_tree.heading('write_bytes', text='Write Bytes')
        self.process_tree.heading('bound', text='Bound')
        self.process_tree.heading('burst_duration', text='Burst Duration')
        self.process_tree.pack(expand=True, fill=tk.BOTH)
        self.cpu_usage_history = {}
        self.update_thread = threading.Thread(target=self.update_process_info_thread, daemon=True)
        self.update_thread.start()

    def update_process_info_thread(self):
        while True:
            process_data = self.get_process_info()
            self.root.after(1000, self.update_gui, process_data)



    def get_process_info(self):
        process_data = []
        for proc in psutil.process_iter(['pid', 'name', 'username', 'cpu_percent', 'memory_percent']):
            process_info = gather_process_info(proc)
            if process_info:
                # Update CPU usage history
                pid = process_info['pid']
                if pid in self.cpu_usage_history:
                    self.cpu_usage_history[pid].append(process_info['cpu_percent'])
                else:
                    self.cpu_usage_history[pid] = [process_info['cpu_percent']]

                process_data.append(process_info)

        return process_data

    def update_gui(self, process_data):
        self.process_tree.delete(*self.process_tree.get_children())
        for info in process_data:
            self.process_tree.insert('', 'end', values=(info['pid'], info['name'], info['cpu_percent'],
                                                        info['memory_percent'], info['read_bytes'], info['write_bytes'],
                                                        info['bound'], info['burst_duration']))



    def run_experiment(self, scheduler):
        start_time = time.time()
        scheduler.round_robin(2)
        elapsed_time_rr = time.time() - start_time
        fairness_rr = scheduler.calculate_fairness()
        throughput_rr = scheduler.calculate_throughput(elapsed_time_rr)
        response_time_rr = scheduler.calculate_response_time(elapsed_time_rr)
        print("Round Robin:")
        print(f"Fairness: {fairness_rr}")
        print(f"Throughput: {throughput_rr}")
        print(f"Response Time: {response_time_rr}")

        start_time = time.time()
        scheduler.first_come_first_serve()
        elapsed_time_fcfs = time.time() - start_time
        fairness_fcfs = scheduler.calculate_fairness()
        throughput_fcfs = scheduler.calculate_throughput(elapsed_time_fcfs)
        response_time_fcfs = scheduler.calculate_response_time(elapsed_time_fcfs)
        print("\nFirst Come First Serve:")
        print(f"Fairness: {fairness_fcfs}")
        print(f"Throughput: {throughput_fcfs}")
        print(f"Response Time: {response_time_fcfs}")

        start_time = time.time()
        scheduler.shortest_job_first()
        elapsed_time_sjf = time.time() - start_time
        fairness_sjf = scheduler.calculate_fairness()
        throughput_sjf = scheduler.calculate_throughput(elapsed_time_sjf)
        response_time_sjf = scheduler.calculate_response_time(elapsed_time_sjf)
        print("\nShortest Job First:")
        print(f"Fairness: {fairness_sjf}")
        print(f"Throughput: {throughput_sjf}")
        print(f"Response Time: {response_time_sjf}")
        throughput_data = [throughput_rr, throughput_fcfs, throughput_sjf]
        response_time_data = [response_time_rr, response_time_fcfs, response_time_sjf]

        self.plot_metrics(throughput_data, response_time_data)


    def plot_metrics(self, throughput_data, response_time_data):
        plt.figure(figsize=(10, 6))

        # Plot throughput
        plt.subplot(2, 1, 1)
        plt.bar(['RR', 'FCFS', 'SJF'], throughput_data, color=['blue', 'green', 'red'])
        plt.title('Throughput of Scheduling Algorithms')
        plt.ylabel('Throughput')

        # Plot response time
        plt.subplot(2, 1, 2)
        plt.bar(['RR', 'FCFS', 'SJF'], response_time_data, color=['blue', 'green', 'red'])
        plt.title('Response Time of Scheduling Algorithms')
        plt.ylabel('Response Time')

        plt.tight_layout()
        plt.show()

def main():
    user_processes, system_processes = categorize_processes()

    print("User Processes:")
    for proc in user_processes:
        print(f"PID: {proc['pid']}, Name: {proc['name'] or 'N/A'}, User: {proc.get('username', 'N/A')}")

    print("\nSystem Processes:")
    for proc in system_processes:
        print(f"PID: {proc['pid']}, Name: {proc['name'] or 'N/A'}, User: {proc.get('username', 'N/A')}")

    user_process_info, system_process_info = gather_utilization_info()

    print("\nUser Process Utilization Information:")
    for info in user_process_info:
        print(f"PID: {info['pid']}, Name: {info['name']}, CPU %: {info['cpu_percent']}, Memory %: {info['memory_percent']}, Read Bytes: {info['read_bytes']}, Write Bytes: {info['write_bytes']}, Bound: {info['bound']}, Burst Duration: {info['burst_duration']}")

    print("\nSystem Process Utilization Information:")
    for info in system_process_info:
        print(f"PID: {info['pid']}, Name: {info['name']}, CPU %: {info['cpu_percent']}, Memory %: {info['memory_percent']}, Read Bytes: {info['read_bytes']}, Write Bytes: {info['write_bytes']}, Bound: {info['bound']}, Burst Duration: {info['burst_duration']}")

    root = tk.Tk()
    app = ProcessMonitor(root)

    scheduler = Scheduler(user_process_info)
    app.run_experiment(scheduler)

    root.mainloop()

if __name__ == "__main__":
    main()
