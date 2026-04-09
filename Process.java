import java.util.*;

class Process {
    int id;
    int load;
    List<Integer> tasks;

    public Process(int id) {
        this.id = id;
        this.load = 0;
        this.tasks = new ArrayList<>();
    }

    public void assignTask(int time) {
        tasks.add(time);
        load += time;
    }
}

public class LoadBalancing {

    public static void main(String[] args) {

        List<Process> processes = new ArrayList<>();
        processes.add(new Process(1));
        processes.add(new Process(2));
        processes.add(new Process(3));

        int[] tasks = {10, 5, 7, 3, 4, 3};

        for (int task : tasks) {

            Process minProcess = processes.get(0);

            for (Process p : processes) {
                if (p.load < minProcess.load) {
                    minProcess = p;
                }
            }

            minProcess.assignTask(task);
        }

        for (Process p : processes) {
            System.out.print("P" + p.id + " (");
            
            for (int i = 0; i < p.tasks.size(); i++) {
                System.out.print(p.tasks.get(i));
                if (i != p.tasks.size() - 1) {
                    System.out.print(", ");
                }
            }
            
            System.out.println(")");
        }
    }
}