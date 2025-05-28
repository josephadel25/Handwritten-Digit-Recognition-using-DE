from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy('float32')
from datetime import datetime
import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import sys
from PyQt5.QtWidgets import (
    QMainWindow, QApplication, QHBoxLayout, QWidget, QVBoxLayout,
    QLabel, QLineEdit, QPushButton, QSizePolicy, QTextEdit, QComboBox
)
from PyQt5.QtCore import Qt
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from de_algorithms import (
    gen_population, differential_evolution,
    mutation_rand_1, mutation_best_1,
    crossover_binomial, crossover_exponential,
    select_better, select_tournament,
    selection_crowding,set_seed
)
from model_utils import build_model, set_model_weights,load_mnist_data
import os
with tf.device('/GPU:0'):

    x_train, y_train, x_val, y_val, x_test, y_test = load_mnist_data()
    x_val = np.asarray(x_val)
    y_val = np.asarray(y_val)

    class EAWithGUI(QMainWindow):
        def __init__(self):
            super().__init__()
            self.setWindowTitle('Differential Evolution for MNIST')
            self.setWindowState(Qt.WindowMaximized)
            self.setStyleSheet("background-color: #1E0342; color: #FFFFFF;")

            self.input_fields = []
            self.fitness_curve = []
            self.accuracy_label = None
            self.output_text = None

            self.fig = plt.figure(figsize=(10, 10))
            self.fig.patch.set_facecolor('#000000') 
            gs = GridSpec(3, 2, hspace=0.5, wspace=0.3)
            # Fitness curve   
            self.ax = self.fig.add_subplot(gs[0, 0])
            self.ax.set_facecolor('#000000')
            self.ax.tick_params(colors='white')
            self.ax.spines[:].set_color('white')
            # Accuracy comparison
            self.ax_comparison = self.fig.add_subplot(gs[0, 1])
            self.ax_comparison.set_facecolor('#000000')
            self.ax_comparison.tick_params(colors='white')
            self.ax_comparison.spines[:].set_color('white')
            # Confusion matrix
            self.ax_confmat = self.fig.add_subplot(gs[1, 0])
            self.ax_confmat.set_facecolor('#000000')
            self.ax_confmat.tick_params(colors='white')
            self.ax_confmat.spines[:].set_color('white')
            # Final accuracy bar plot
            self.ax_bar = self.fig.add_subplot(gs[1, 1])
            self.ax_bar.set_facecolor('#000000')
            self.ax_bar.tick_params(colors='white')
            self.ax_bar.spines[:].set_color('white')
            # Sample predictions
            self.ax_samples = self.fig.add_subplot(gs[2, :])
            self.ax_samples.set_facecolor('#000000')
            self.ax_samples.tick_params(colors='white')
            self.ax_samples.spines[:].set_color('white')

            self.canvas = FigureCanvas(self.fig)
            self.init_gui()

        def init_gui(self):
            main_layout = QHBoxLayout()
            # left panel
            left_panel = QWidget()
            inputs_layout = QVBoxLayout(left_panel)
            inputs_layout.setSpacing(10)
            inputs_layout.setContentsMargins(20, 30, 20, 20)
            # seed input
            seed_label = QLabel("Random Seed")
            seed_label.setStyleSheet("color: #00e5ff; font-weight: bold;")
            self.seed_field = QLineEdit()
            self.seed_field.setStyleSheet(""" QLineEdit { background-color: #000000; color: #00e5ff; border: 1px solid #00e5ff; border-radius: 4px; padding: 6px; } """)
            inputs_layout.addWidget(seed_label)
            inputs_layout.addWidget(self.seed_field)
            # other inputs
            labels = ['Population Size', 'Mutation Factor (F)', 'Crossover Rate (CR)', 'Generations']
            for label in labels:
                lbl = QLabel(label)
                lbl.setStyleSheet("color: #00e5ff; font-weight: bold;")
                field = QLineEdit()
                field.setStyleSheet(""" QLineEdit { background-color: #000000; color: #00e5ff; border: 1px solid #00e5ff; border-radius: 4px; padding: 6px; } """)
                inputs_layout.addWidget(lbl)
                inputs_layout.addWidget(field)
                self.input_fields.append(field)

            # mutation
            self.mutation_dropdown = QComboBox()
            self.mutation_dropdown.addItems(["mutation_rand_1", "mutation_best_1"])
            self.mutation_dropdown.setStyleSheet("background-color: #000000; color: #00e5ff;border: 1px solid #00e5ff;border-radius: 4px;padding: 6px;")
            inputs_layout.addWidget(QLabel("Mutation Strategy", styleSheet="color: #00e5ff; font-weight: bold;"))
            inputs_layout.addWidget(self.mutation_dropdown)
            # crossover
            self.crossover_dropdown = QComboBox()
            self.crossover_dropdown.addItems(["crossover_binomial", "crossover_exponential"])
            self.crossover_dropdown.setStyleSheet("background-color: #000000; color: #00e5ff;border: 1px solid #00e5ff;border-radius: 4px;padding: 6px;")
            inputs_layout.addWidget(QLabel("Crossover Strategy", styleSheet="color: #00e5ff; font-weight: bold;"))
            inputs_layout.addWidget(self.crossover_dropdown)
            # selection
            self.selection_dropdown = QComboBox()
            self.selection_dropdown.addItems(["select_better", "select_tournament", "Crowding"])
            self.selection_dropdown.setStyleSheet("background-color: #000000; color: #00e5ff;border: 1px solid #00e5ff;border-radius: 4px;padding: 6px;")
            inputs_layout.addWidget(QLabel("Selection Strategy", styleSheet="color: #00e5ff; font-weight: bold;"))
            inputs_layout.addWidget(self.selection_dropdown)
            # initialization
            self.population_dropdown = QComboBox()
            self.population_dropdown.addItems(["random", "gaussian"]) 
            self.population_dropdown.setStyleSheet("background-color: #000000; color: #00e5ff;border: 1px solid #00e5ff;border-radius: 4px;padding: 6px;")
            inputs_layout.addWidget(QLabel("Population Initialization Strategy", styleSheet="color: #00e5ff; font-weight: bold;"))
            inputs_layout.addWidget(self.population_dropdown)
            # accuracy label
            self.accuracy_label = QLabel("Best Accuracy:     ")
            self.accuracy_label.setStyleSheet("color: #00e5ff; font-weight: bold; font-size: 20px;padding: 30px;padding-left:1px;")
            inputs_layout.addWidget(self.accuracy_label)
            # output log
            self.output_text = QTextEdit()
            self.output_text.setReadOnly(True)
            self.output_text.setStyleSheet(""" QTextEdit { background-color: #000000; color: #00e5ff; font-size: 18px; border: 1px solid #00e5ff; padding: 10px; font-family: Consolas; } """)
            self.output_text.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            inputs_layout.addWidget(self.output_text, 4)
            # run button
            run_btn = QPushButton("Run DE")
            run_btn.setStyleSheet(""" QPushButton { background-color: #00e5ff; color: #000000; font-weight: bold; padding: 8px; border-radius: 5px; } QPushButton:hover { background-color: #33ffff; } """)
            inputs_layout.addStretch(1)
            inputs_layout.addWidget(run_btn)
            run_btn.clicked.connect(self.run_de)
            # save btton
            save_btn = QPushButton("Save Results")
            save_btn.setStyleSheet("""QPushButton { background-color: #ffaa00; color: #000000; font-weight: bold; padding: 8px; border-radius: 5px; } QPushButton:hover { background-color: #ffcc33; } """)
            inputs_layout.addWidget(save_btn)
            save_btn.clicked.connect(self.save_results)
            main_layout.addWidget(left_panel)
            # right panel
            right_panel = QWidget()
            output_layout = QVBoxLayout(right_panel)
            output_layout.addWidget(self.canvas)
            main_layout.addWidget(right_panel)
            container = QWidget()
            container.setLayout(main_layout)
            self.setCentralWidget(container)

        def save_results(self):
            # make the dirictory
            best_acc = self.accuracy_label.text().split(":")[1].strip()
            output_dir = f"results/{best_acc}"
            os.makedirs(output_dir, exist_ok=True)
            # save figurs
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            plot_path = os.path.join(output_dir, f"fitness_comparison_{timestamp}.png")
            self.fig.savefig(plot_path, facecolor=self.fig.get_facecolor())
            # save parameters in file
            param_path = os.path.join(output_dir, f"parameters_{timestamp}.txt")
            with open(param_path, 'w') as f:
                f.write("=== Selected Parameters ===\n")
                f.write(f"Optimization Algorithm : Differential Evolution \n")
                f.write(f"Random Seed: {self.seed_field.text()}\n")
                f.write(f"Population Size: {self.input_fields[0].text()}\n")
                f.write(f"Mutation Factor (F): {self.input_fields[1].text()}\n")
                f.write(f"Crossover Rate (CR): {self.input_fields[2].text()}\n")
                f.write(f"Generations: {self.input_fields[3].text()}\n")
                f.write(f"Mutation Strategy: {self.mutation_dropdown.currentText()}\n")
                f.write(f"Crossover Strategy: {self.crossover_dropdown.currentText()}\n")
                f.write(f"Selection Strategy: {self.selection_dropdown.currentText()}\n")
                f.write(f"Population Init Strategy: {self.population_dropdown.currentText()}\n")
                f.write(f"Final Accuracy (on test): {best_acc}\n\n")
                f.write("=== Output Log ===\n")
                f.write(self.output_text.toPlainText())
            # generations accuracy
            current_time = datetime.now().strftime('%H:%M:%S') 
            gen_accuracy_path = os.path.join(output_dir, f"gen_accuracies_{timestamp}.txt")
            with open(gen_accuracy_path, 'w') as f:
                f.write("=== Generation-wise Accuracies ===\n")
                for gen, acc in enumerate(self.fitness_curve):
                    f.write(f"[{current_time}] Generation {gen+1}: Accuracy = {acc:.4f}\n")
            self.output_text.append(f"\nSaved parameters and plots to 'results/{best_acc}' as:\n- {plot_path}\n- {param_path}\n- {gen_accuracy_path}")

        def run_de(self):
            try:
                seed = int(self.seed_field.text())
                set_seed(seed) 
            except ValueError:
                self.output_text.append("Invalid seed value. Please enter a valid integer.")
                return
            try:
                pop_size = int(self.input_fields[0].text())
                F = float(self.input_fields[1].text())  
                CR = float(self.input_fields[2].text())           
                generations = int(self.input_fields[3].text())
            except ValueError:
                self.output_text.append("One or more parameters are invalid. Please enter numeric values.")
                return

            mutation_map = {
                "mutation_rand_1": mutation_rand_1,
                "mutation_best_1": mutation_best_1
            }
            crossover_map = {
                "crossover_binomial": crossover_binomial,
                "crossover_exponential": crossover_exponential
            }
            select_map = {
                "select_better": select_better,
                "select_tournament": select_tournament,
                "Crowding": selection_crowding
            }
            init_map = {
                "random": "random",
                "gaussian": "gaussian"
            }
            # the choosen operators
            mutation_func = mutation_map[self.mutation_dropdown.currentText()]
            crossover_func = crossover_map[self.crossover_dropdown.currentText()]
            selection_func = select_map[self.selection_dropdown.currentText()]
            init_func = init_map[self.population_dropdown.currentText()]
            # take a subset of the val
            DE_subset_ratio = 0.2
            subset_size = int(DE_subset_ratio * len(x_val)) 
            x_val_sub = x_val[:subset_size]  
            y_val_sub = y_val[:subset_size]  

            global_model = build_model()
            global_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

            population, fitnesses = gen_population(pop_size, global_model, x_val_sub, y_val_sub, strategy=init_func)

            self.fitness_curve = []
            def update_progress(gen, acc):
                self.fitness_curve.append(acc)
                current_time = datetime.now().strftime('%H:%M:%S') 
                self.output_text.append(f"[{current_time}] Generation {gen}: Accuracy = {acc:.4f}")
                QApplication.processEvents()

            best_solution, fitness_curve = differential_evolution(
                global_model,population, fitnesses, 
                generations,mutation_func=mutation_func,
                crossover_func=crossover_func,
                selection_func=selection_func,
                F=F, CR=CR, x_val=x_val_sub, y_val=y_val_sub,
                progress_callback=update_progress
            )

            model = build_model(seed=seed)
            set_model_weights(model, best_solution)
            model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
            self.accuracy_label.setText(f"Best Accuracy: {fitness_curve[-1]:.4f}")

            self.ax.clear()
            self.ax.set_facecolor('#000000')
            self.fig.patch.set_facecolor('#000000')
            self.ax.plot(fitness_curve, color='#00e5ff', label='Fitness')
            self.ax.set_title('Fitness Curve', color='white')
            self.ax.set_xlabel('Generation', color='white')
            self.ax.set_ylabel('Accuracy', color='white')
            self.ax.tick_params(colors='white')
            self.ax.spines[:].set_color('white')
            self.ax.legend(facecolor='#000000', edgecolor='white', labelcolor='white')
            # Fine-tuned model with DE
            history_de = model.fit(x_train, y_train, epochs=5, batch_size=128, validation_data=(x_val, y_val))
            _, acc_ft = model.evaluate(x_test, y_test, verbose=0)
            self.output_text.append(f"Fine-tuned Accuracy: {acc_ft:.4f}")
            # Backprop model
            backprop_model = build_model(seed=seed)
            backprop_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
            history_bp = backprop_model.fit(x_train, y_train, epochs=5, validation_data=(x_val, y_val))
            _, test_acc_bp = backprop_model.evaluate(x_test, y_test, verbose=0)
            self.output_text.append(f"Backprop Only Accuracy: {test_acc_bp:.4f}")
            # Confusion Matrix
            y_pred = model.predict(x_test)
            y_pred_classes = np.argmax(y_pred, axis=1)
            y_true = np.argmax(y_test, axis=1)
            conf_mat = confusion_matrix(y_true, y_pred_classes)
            self.ax_confmat.clear()
            self.ax_confmat.set_facecolor('#000000')
            disp = ConfusionMatrixDisplay(confusion_matrix=conf_mat)
            disp.plot(ax=self.ax_confmat, cmap='Blues', colorbar=False)
            self.ax_confmat.set_aspect('auto', adjustable='box')
            for label in self.ax_confmat.get_xticklabels() + self.ax_confmat.get_yticklabels():
                label.set_color('white')
                label.set_fontsize(14)
            self.ax_confmat.set_title('Confusion Matrix (DE Model)', color='white', fontsize=8)
            # Comparison of Training Accuracy
            self.ax_comparison.clear()
            self.ax_comparison.set_facecolor('#000000')
            self.fig.patch.set_facecolor('#000000')
            self.ax_comparison.plot(history_de.history['accuracy'], label='DE Model', color='#00e5ff')
            self.ax_comparison.plot(history_bp.history['accuracy'], label='BP Model', color='#ff6347')
            self.ax_comparison.set_title('Comparison of Training Accuracy', color='white')
            self.ax_comparison.set_xlabel('Epochs', color='white')
            self.ax_comparison.set_ylabel('Accuracy', color='white')
            self.ax_comparison.tick_params(colors='white')
            self.ax_comparison.spines[:].set_color('white')
            self.ax_comparison.legend(facecolor='#000000', edgecolor='white', labelcolor='white')
            self.canvas.draw()
            # Final Test Accuracy Comparison
            self.ax_bar.clear()
            self.ax_bar.set_facecolor('#000000')
            self.fig.patch.set_facecolor('#000000')
            models = ['DE', 'Backprop']
            accuracies = [fitness_curve[-1], test_acc_bp]
            bar_colors = ['#00e5ff', '#ff6347']
            bars = self.ax_bar.bar(models, accuracies, color=bar_colors)
            self.ax_bar.set_ylim(0, 1 )
            self.ax_bar.set_ylabel('Test Accuracy', color='white')
            self.ax_bar.set_title('Final Test Accuracy Comparison', color='white')
            self.ax_bar.tick_params(colors='white')
            self.ax_bar.spines[:].set_color('white')
            for bar in bars:
                height = bar.get_height()
                self.ax_bar.annotate(f'{height:.4f}',xy=(bar.get_x() + bar.get_width() / 2, height),xytext=(0, 3),textcoords="offset points",ha='center', va='bottom',color='white')
            # sampels predections
            self.ax_samples.clear()
            self.ax_samples.set_facecolor('#000000')
            for spine in self.ax_samples.spines.values():
                spine.set_color('black')
            self.ax_samples.tick_params(colors='black')
            sample_indices = np.random.choice(len(x_test), size=10, replace=False)
            for i, idx in enumerate(sample_indices):
                image = x_test[idx].reshape(28, 28)
                true_label = y_true[idx]
                pred_label = y_pred_classes[idx]
                ax_img = self.ax_samples.inset_axes([i * 0.1, 0.05, 0.09, 0.9])
                ax_img.imshow(image, cmap='gray')
                ax_img.set_xticks([])
                ax_img.set_yticks([])
                ax_img.set_title(f"T:{true_label}\nP:{pred_label}", fontsize=8, color='white')
            self.ax_samples.set_title("Sample Predictions (T=True, P=Pred)", color='white')
            self.canvas.draw()


    if __name__ == '__main__':
        app = QApplication(sys.argv)
        window = EAWithGUI()
        window.resize(1000, 600)
        window.show()
        sys.exit(app.exec_())
