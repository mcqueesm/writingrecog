export class Network {
  constructor(sizes) {
    this.num_layers = sizes.length;
    this.sizes = sizes;
    this.biases = [];
    for (let i = 1; i < sizes.length; i++) {
      this.biases.push(tf.randomNormal([sizes[i], 1]));
    }
    this.weights = [];
    for (let j = 0; j < sizes.length - 1; j++) {
      this.weights.push(tf.randomNormal([sizes[j + 1], sizes[j]]));
    }
  }
  shuffleArray(array) {
    for (let i = array.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [array[i], array[j]] = [array[j], array[i]];
    }
  }
  feedforward(act) {
    let a = act;
    for (let i = 0; i < this.num_layers - 1; i++) {
      a = tf.tidy(() => tf.sigmoid(this.weights[i].dot(a).add(this.biases[i])));
    }
    return a;
  }

  SGD(training_data, epochs, mini_batch_size, eta, test_data = null) {
    let n_test;
    let n = training_data.length;
    if (test_data) n_test = test_data.length;
    for (let j = 0; j < epochs; j++) {
      this.shuffleArray(training_data);
      let mini_batches = [];
      for (let k = 0; k < n; k += mini_batch_size) {
        mini_batches.push(training_data.slice(k, k + mini_batch_size));
      }
      mini_batches.forEach(mb => {
        [this.weights, this.biases] = tf.tidy(() =>
          this.update_mini_batch([...mb], eta)
        );
      });
      localStorage.setItem(
        "weights",
        JSON.stringify(this.weights.map(x => x.arraySync()))
      );
      localStorage.setItem(
        "biases",
        JSON.stringify(this.biases.map(x => x.arraySync()))
      );
      if (test_data) {
        console.log(`Epoch ${j}: ${this.evaluate(test_data)} / ${n_test}`);
      } else {
        console.log(`Epoch ${j} complete`);
      }
      console.log("Epoch complete:");
      console.log("Weights:");
      this.weights.forEach(x => x.print());
      console.log("Biases:");
      this.biases.forEach(x => x.print());
    }
  }

  update_mini_batch(mini_batch, eta) {
    //console.log(tf.memory().numTensors);
    let nabla_b = [];
    let nabla_w = [];
    for (let i = 0; i < this.num_layers - 1; i++) {
      nabla_b.push(tf.zeros(this.biases[i].shape));
      nabla_w.push(tf.zeros(this.weights[i].shape));
    }
    let x, y;
    mini_batch.forEach(data => {
      x = data[0];
      y = data[1];
      let delta_nabla_b, delta_nabla_w;
      [delta_nabla_b, delta_nabla_w] = this.backprop(x, y);
      nabla_b = nabla_b.map((nb, i) => {
        return nb.add(delta_nabla_b[i]);
      });
      nabla_w = nabla_w.map((nw, i) => {
        return nw.add(delta_nabla_w[i]);
      });
    });

    let weights = this.weights.map((w, i) => {
      return w.sub(tf.mul(nabla_w[i], eta / mini_batch.length));
    });
    let biases = this.biases.map((b, i) => {
      return b.sub(tf.mul(nabla_b[i], eta / mini_batch.length));
    });
    this.weights.forEach((x, i) => {
      x.dispose();
      this.biases[i].dispose();
    });

    return [weights, biases];
  }
  backprop(x, y) {
    let nabla_b = [];
    let nabla_w = [];
    for (let i = 0; i < this.num_layers - 1; i++) {
      nabla_b.push(tf.zeros(this.biases[i].shape));
      nabla_w.push(tf.zeros(this.weights[i].shape));
    }
    let activation = x;
    let activations = [x];
    let zs = [];
    this.biases.forEach((b, i) => {
      let z = this.weights[i].dot(activation).add(b);
      zs.push(z);
      activation = z.sigmoid();
      activations.push(activation);
    });
    let delta = this.cost_derivative(
      activations[activations.length - 1],
      y
    ).mul(this.sigmoid_prime(zs[zs.length - 1]));
    nabla_b[nabla_b.length - 1] = delta;
    nabla_w[nabla_w.length - 1] = delta.dot(
      activations[activations.length - 2].transpose()
    );
    for (let i = this.num_layers - 2; i > 0; i--) {
      let z = zs[i - 1];
      let sp = this.sigmoid_prime(z);
      delta = this.weights[i]
        .transpose()
        .dot(delta)
        .mul(sp);
      nabla_b[i - 1] = delta;
      nabla_w[i - 1] = delta.dot(activations[i - 1].transpose());

      //sp.dispose();
    }

    return [nabla_b, nabla_w];
  }
  evaluate(test_data) {
    let sum = 0;
    test_data.forEach(data => {
      let x = tf.tidy(() => this.feedforward(data[0]).argMax());
      let y = data[1].argMax();

      let xvalue = x.dataSync()[0];
      let yvalue = y.dataSync()[0];

      if (xvalue === yvalue) {
        sum++;
      }
      x.dispose();
    });
    return sum;
  }
  cost_derivative(output_activations, y) {
    return output_activations.sub(y);
  }
  sigmoid_prime(z) {
    return z.sigmoid().mul(tf.sub(1, z.sigmoid()));
  }

  setWeights(weights) {
    let newWeights = [];
    weights.forEach(x =>
      newWeights.push(tf.tensor(x, [x.length, x[0].length]))
    );
    this.weights = newWeights;
  }

  setBiases(biases) {
    let newBiases = [];
    biases.forEach(x => newBiases.push(tf.tensor(x, [x.length, 1])));
    this.biases = newBiases;
  }
}
