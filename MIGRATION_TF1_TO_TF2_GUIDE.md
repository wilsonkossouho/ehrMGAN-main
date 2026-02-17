# 🔄 Guide Complet : Migration TensorFlow 1.x → 2.x

**Projet :** EHR-M-GAN
**Version actuelle :** TensorFlow 1.15
**Version cible :** TensorFlow 2.10+
**Difficulté :** ⭐⭐⭐⭐ (Élevée - refactorisation majeure)

---

## 📋 TABLE DES MATIÈRES

1. [Vue d'ensemble de la migration](#vue-densemble)
2. [Étape 0 : Préparation](#étape-0-préparation)
3. [Étape 1 : Analyse des dépendances TF1](#étape-1-analyse-tf1)
4. [Étape 2 : Migration des imports](#étape-2-imports)
5. [Étape 3 : Remplacer Session par Eager Execution](#étape-3-sessions)
6. [Étape 4 : Migrer les RNN/LSTM](#étape-4-rnn)
7. [Étape 5 : Migrer les losses et optimizers](#étape-5-losses)
8. [Étape 6 : Remplacer Saver par Checkpoint](#étape-6-checkpoint)
9. [Étape 7 : Tests et validation](#étape-7-tests)
10. [Checklist finale](#checklist-finale)

---

## 🎯 VUE D'ENSEMBLE

### Pourquoi migrer ?

**Problèmes actuels avec TF1.x** :
- ❌ Plus de support officiel depuis 2021
- ❌ Incompatible avec Python 3.8+
- ❌ Pas d'accès aux nouvelles fonctionnalités TF2
- ❌ Difficile à déployer (Docker, cloud)
- ❌ Performance inférieure (pas de mixed precision native)

**Avantages TF2** :
- ✅ Eager execution (debug plus facile)
- ✅ API Keras intégrée (code plus court)
- ✅ @tf.function (compilation automatique)
- ✅ Support Python 3.12
- ✅ Mixed precision native (gain 2-3x vitesse)
- ✅ Distribution multi-GPU simplifiée

### Changements majeurs TF1 → TF2

| Concept TF1 | Équivalent TF2 | Difficulté |
|-------------|----------------|------------|
| `tf.Session()` | Eager execution / `@tf.function` | ⭐⭐⭐⭐ |
| `tf.placeholder` | Arguments de fonction | ⭐⭐⭐ |
| `tf.variable_scope` | Layers Keras / `tf.Module` | ⭐⭐⭐⭐ |
| `tf.contrib.*` | Supprimé (alternatives variées) | ⭐⭐⭐⭐⭐ |
| `tf.train.Saver` | `tf.train.Checkpoint` | ⭐⭐ |
| `tf.layers.dense` | `tf.keras.layers.Dense` | ⭐ |
| `tf.nn.rnn_cell` | `tf.keras.layers.LSTM` | ⭐⭐⭐ |

### Temps estimé

- **Migration rapide (tf_upgrade_v2)** : 2-4 heures + tests
- **Migration manuelle complète** : 20-40 heures
- **Optimisation TF2 native** : +10-20 heures

Nous allons suivre une approche **hybride** :
1. Utiliser `tf_upgrade_v2` pour conversion automatique (70%)
2. Refactoriser manuellement les parties critiques (30%)

---

## 📦 ÉTAPE 0 : PRÉPARATION

### 0.1 Sauvegarder le code actuel

```bash
# Créer une branche de sauvegarde
git checkout -b backup-tf1-original
git add -A
git commit -m "Backup: Code TF1.15 fonctionnel avant migration"
git push origin backup-tf1-original

# Créer branche de migration
git checkout master
git checkout -b feature/migrate-to-tf2
```

### 0.2 Installer outils de migration

```bash
# Environnement Python 3.10+
python -m venv venv_tf2
source venv_tf2/bin/activate  # Windows: venv_tf2\Scripts\activate

# Installer TensorFlow 2 + outils
pip install tensorflow==2.10.1
pip install tf-nightly  # Pour tf_upgrade_v2
pip install tensorflow-datasets
pip install keras==2.10.0
```

### 0.3 Créer structure de migration

```bash
mkdir migration_tf2
cd migration_tf2

# Copier fichiers originaux
cp ../main_train.py ./main_train_tf1.py
cp ../m3gan.py ./m3gan_tf1.py
cp ../networks.py ./networks_tf1.py
cp ../Bilateral_lstm_class.py ./Bilateral_lstm_class_tf1.py
```

---

## 🔍 ÉTAPE 1 : ANALYSE DES DÉPENDANCES TF1

### 1.1 Identifier tous les usages TF1

```bash
# Lister tous les fichiers Python
find . -name "*.py" -type f > python_files.txt

# Rechercher patterns TF1 critiques
grep -r "tf.Session" . --include="*.py"
grep -r "tf.placeholder" . --include="*.py"
grep -r "tf.contrib" . --include="*.py"
grep -r "tf.AUTO_REUSE" . --include="*.py"
grep -r "tf.variable_scope" . --include="*.py"
grep -r "tf.train.Saver" . --include="*.py"
```

### 1.2 Résultats de l'analyse

**Fichiers à migrer (par ordre de priorité)** :

| Fichier | Usages TF1 | Complexité | Priorité |
|---------|------------|------------|----------|
| `main_train.py` | Session, reset_default_graph, ConfigProto | ⭐⭐⭐ | 1 |
| `m3gan.py` | Session, Saver, placeholders, train ops | ⭐⭐⭐⭐ | 1 |
| `networks.py` | contrib, variable_scope, AUTO_REUSE, RNN cells | ⭐⭐⭐⭐⭐ | 1 |
| `Bilateral_lstm_class.py` | RNNCell custom, variable_scope | ⭐⭐⭐⭐ | 2 |
| `init_state.py` | RNN states | ⭐⭐ | 3 |
| `evaluation_metrics/*.py` | Placeholders (discriminative_score) | ⭐⭐ | 3 |

### 1.3 Dépendances problématiques

**Critique** :
```python
# networks.py ligne 4
from tensorflow.contrib.layers import l2_regularizer
# ❌ tensorflow.contrib SUPPRIMÉ dans TF2
# ✅ Remplacer par : tf.keras.regularizers.l2()
```

**Custom LSTM** :
```python
# Bilateral_lstm_class.py
# Implémentation custom de RNNCell
# ⚠️ Nécessite refactorisation complète en tf.keras.layers.Layer
```

---

## 🔧 ÉTAPE 2 : MIGRATION DES IMPORTS

### 2.1 Utiliser tf_upgrade_v2

```bash
# Conversion automatique
tf_upgrade_v2 \
  --intree ./ \
  --outtree ../migration_tf2_auto \
  --reportfile migration_report.txt

# Lire le rapport
cat migration_report.txt
```

**Attendez-vous à voir** :
- ✅ Conversions automatiques : ~60-70%
- ⚠️ Avertissements : ~20-30%
- ❌ Erreurs manuelles requises : ~10%

### 2.2 Remplacements manuels (networks.py)

**AVANT (TF1)** :
```python
import tensorflow as tf
from tensorflow.contrib.layers import l2_regularizer
```

**APRÈS (TF2)** :
```python
import tensorflow as tf
from tensorflow.keras.regularizers import l2 as l2_regularizer

# Alternative si conflit :
# l2_reg = tf.keras.regularizers.L2(l2=0.001)
```

### 2.3 Table de correspondance imports

| TF1 | TF2 | Notes |
|-----|-----|-------|
| `tf.contrib.layers.l2_regularizer` | `tf.keras.regularizers.l2()` | Syntaxe légèrement différente |
| `tf.nn.rnn_cell.LSTMCell` | `tf.keras.layers.LSTMCell` | Préférer `tf.keras.layers.LSTM` |
| `tf.nn.rnn_cell.MultiRNNCell` | `tf.keras.layers.StackedRNNCells` | Ou stack de Layers |
| `tf.layers.dense` | `tf.keras.layers.Dense` | API identique |
| `tf.train.AdamOptimizer` | `tf.keras.optimizers.Adam` | API identique |

### 2.4 Créer fichier de migration

Créer `migration_tf2/imports_tf2.py` :

```python
"""
Imports TF2 consolidés pour EHR-M-GAN
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers, optimizers
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.regularizers import l2

# Raccourcis pour compatibilité
l2_regularizer = l2

# Vérifier version
assert tf.__version__.startswith('2.'), f"TensorFlow 2.x requis, version actuelle : {tf.__version__}"

print(f"✅ TensorFlow {tf.__version__} importé avec succès")
```

---

## 🚀 ÉTAPE 3 : REMPLACER SESSION PAR EAGER EXECUTION

### 3.1 Comprendre la différence

**TF1 (Graph mode)** :
```python
# Construction du graphe
x = tf.placeholder(tf.float32, [None, 10])
y = tf.matmul(x, W)

# Exécution dans session
with tf.Session() as sess:
    result = sess.run(y, feed_dict={x: data})
```

**TF2 (Eager mode)** :
```python
# Exécution immédiate
x = tf.constant(data)  # Pas de placeholder
y = tf.matmul(x, W)
# 'y' est déjà calculé, pas besoin de sess.run()
print(y.numpy())  # Conversion directe en NumPy
```

### 3.2 Migrer main_train.py

**AVANT (TF1)** - `main_train.py` lignes 70-100 :
```python
tf.reset_default_graph()
run_config = tf.ConfigProto()
with tf.Session(config=run_config) as sess:
    model = m3gan(sess=sess, ...)
    model.build()
    model.train()
    d_gen_data, c_gen_data = model.generate_data(num_sample=no_gen)
```

**APRÈS (TF2)** :
```python
# Plus besoin de Session !
model = M3GAN_TF2(  # Notez : plus de 'sess' argument
    batch_size=args.batch_size,
    time_steps=time_steps,
    # ... autres params
)
model.build()
model.train()
d_gen_data, c_gen_data = model.generate_data(num_sample=no_gen)
```

### 3.3 Créer nouvelle classe modèle

Créer `migration_tf2/m3gan_tf2.py` :

```python
import tensorflow as tf
from tensorflow import keras

class M3GAN_TF2(keras.Model):
    """
    Version TF2 de m3gan utilisant tf.keras.Model
    """
    def __init__(self, batch_size, time_steps, **kwargs):
        super(M3GAN_TF2, self).__init__()

        # Plus de 'sess' nécessaire !
        self.batch_size = batch_size
        self.time_steps = time_steps
        # ... copier autres attributs de __init__ original

        # Initialiser sous-modèles (sera détaillé plus tard)
        self.c_vae = None  # À remplacer par version TF2
        self.d_vae = None
        self.c_gan = None
        self.d_gan = None

    def build(self, input_shape=None):
        """
        Remplace l'ancien build_tf_graph()
        """
        # Construction du modèle (eager mode)
        # Plus besoin de placeholders !
        pass

    def call(self, inputs, training=False):
        """
        Forward pass (obligatoire pour keras.Model)
        """
        pass

    def train_step(self, data):
        """
        Une étape d'entraînement (remplace sess.run())
        """
        pass

    @tf.function  # Compilation automatique pour performance
    def generate_data(self, num_sample):
        """
        Génération de données synthétiques
        """
        pass
```

### 3.4 Pattern de migration : Session → Eager

**Pattern général** :

```python
# TF1
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(num_epochs):
        _, loss_val = sess.run([train_op, loss], feed_dict={x: batch_x})

# TF2
# Pas de session, exécution directe
for epoch in range(num_epochs):
    with tf.GradientTape() as tape:
        predictions = model(batch_x, training=True)
        loss_val = loss_fn(batch_y, predictions)

    gradients = tape.gradient(loss_val, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
```

---

## 🧠 ÉTAPE 4 : MIGRER LES RNN/LSTM

### 4.1 Analyser Bilateral_lstm_class.py

**Problème** : Implémentation custom de `tf.nn.rnn_cell.RNNCell` (TF1)

**Solution** : Hériter de `tf.keras.layers.Layer` (TF2)

### 4.2 Migrer Bilateral LSTM Cell

**AVANT (TF1)** - `Bilateral_lstm_class.py` :
```python
from tensorflow.python.ops.rnn_cell_impl import RNNCell

class Bilateral_LSTM_cell(RNNCell):
    def __init__(self, num_units, ...):
        super(Bilateral_LSTM_cell, self).__init__()
        self._num_units = num_units

    @property
    def state_size(self):
        return (self._num_units, self._num_units)

    @property
    def output_size(self):
        return self._num_units

    def __call__(self, inputs, state, scope=None):
        # Logique LSTM custom
        with tf.variable_scope(scope or type(self).__name__):
            # ...
```

**APRÈS (TF2)** :
```python
import tensorflow as tf
from tensorflow.keras.layers import Layer

class BilateralLSTMCell(Layer):
    """
    Version TF2 de Bilateral LSTM Cell
    """
    def __init__(self, num_units, **kwargs):
        super(BilateralLSTMCell, self).__init__(**kwargs)
        self.num_units = num_units
        self.state_size = (num_units, num_units)

    def build(self, input_shape):
        """
        Créer les poids (appelé automatiquement)
        """
        # Remplace variable_scope
        self.W_i = self.add_weight(
            name='W_input',
            shape=(input_shape[-1], self.num_units),
            initializer='glorot_uniform',
            trainable=True
        )
        # ... autres poids

        super().build(input_shape)

    def call(self, inputs, states):
        """
        Remplace __call__
        """
        h_prev, c_prev = states

        # Logique LSTM (identique)
        # ...

        return output, [h_new, c_new]

    def get_config(self):
        """
        Pour sérialisation (optionnel)
        """
        config = super().get_config()
        config.update({'num_units': self.num_units})
        return config
```

### 4.3 Migrer networks.py (VAE)

**AVANT (TF1)** - `networks.py` lignes 30-87 :
```python
def build_vae(self, input_data, conditions=None):
    input_enc = tf.unstack(input_data, axis=1)

    # Créer cellules LSTM
    self.cell_enc = self.buildEncoder()
    self.cell_dec = self.buildDecoder()
    enc_state = self.cell_enc.zero_state(self.batch_size, tf.float32)

    # Loop manuel sur timesteps
    for t in range(self.time_steps):
        with tf.variable_scope('Encoder', regularizer=l2_regularizer(self.l2scale), reuse=tf.AUTO_REUSE):
            h_enc, enc_state = self.cell_enc(tf.concat([input_enc[t], x_hat], 1), enc_state)
        # ...
```

**APRÈS (TF2)** :
```python
class C_VAE_TF2(keras.Model):
    def __init__(self, batch_size, time_steps, dim, z_dim, **kwargs):
        super().__init__()
        self.batch_size = batch_size
        self.time_steps = time_steps
        self.dim = dim
        self.z_dim = z_dim

        # Utiliser Keras Layers
        self.encoder = keras.Sequential([
            keras.layers.LSTM(128, return_sequences=True, return_state=True),
            keras.layers.LSTM(128, return_state=True)
        ])

        self.decoder = keras.layers.LSTM(128, return_sequences=True, return_state=True)

        # Layers de sampling
        self.dense_mu = keras.layers.Dense(z_dim)
        self.dense_sigma = keras.layers.Dense(z_dim)

        # Decoder output
        self.dense_out = keras.layers.Dense(dim, activation='sigmoid')

    @tf.function
    def encode(self, x):
        """Encoder avec return states"""
        # Keras LSTM gère automatiquement les loops temporels !
        _, h_state, c_state = self.encoder(x)
        return h_state, c_state

    @tf.function
    def reparameterize(self, h_enc):
        """Sampling layer"""
        mu = self.dense_mu(h_enc)
        log_sigma = self.dense_sigma(h_enc)
        sigma = tf.exp(log_sigma)

        eps = tf.random.normal(shape=tf.shape(mu))
        z = mu + sigma * eps

        return z, mu, log_sigma

    @tf.function
    def decode(self, z):
        """Decoder"""
        # Répéter z sur timesteps
        z_repeated = tf.tile(tf.expand_dims(z, 1), [1, self.time_steps, 1])

        # LSTM decoder
        h_dec, _, _ = self.decoder(z_repeated)

        # Output
        reconstructed = self.dense_out(h_dec)

        return reconstructed

    def call(self, x, training=False):
        """Forward pass complet"""
        h_enc, _ = self.encode(x)
        z, mu, log_sigma = self.reparameterize(h_enc)
        reconstructed = self.decode(z)

        return reconstructed, mu, log_sigma
```

### 4.4 Alternative : Wrapper TF1 Cell

Si la migration complète est trop complexe, utiliser `tf.compat.v1` temporairement :

```python
import tensorflow.compat.v1 as tf1

# Désactiver eager pour cette partie seulement
@tf.function(experimental_compile=False)
def legacy_bilateral_lstm(inputs):
    # Code TF1 original avec tf1.* au lieu de tf.*
    with tf1.variable_scope('bilateral_lstm'):
        # ... code original
    return outputs

# Utiliser dans modèle TF2
class HybridModel(keras.Model):
    def call(self, x):
        # Partie moderne TF2
        x = self.modern_layer(x)

        # Appeler code legacy
        x = legacy_bilateral_lstm(x)

        return x
```

⚠️ **Attention** : Cette approche est temporaire, prévoir refactorisation complète.

---

## 🎯 ÉTAPE 5 : MIGRER LES LOSSES ET OPTIMIZERS

### 5.1 Identifier les losses actuelles

D'après `m3gan.py`, les losses utilisées :
1. Reconstruction loss (MSE)
2. KL divergence
3. Matching loss (Euclidean distance)
4. Contrastive loss (NT-Xent)
5. Adversarial loss (GAN)
6. Feature matching loss

### 5.2 Migrer les losses

**AVANT (TF1)** :
```python
# m3gan.py - build_loss()
def build_loss(self):
    # Reconstruction loss
    self.c_vae_loss_re = tf.reduce_mean(
        tf.squared_difference(self.c_decoded, self.c_inputs_)
    )

    # KL divergence
    self.c_vae_loss_kl = -0.5 * tf.reduce_mean(
        1 + self.c_logsigma - tf.square(self.c_mu) - tf.exp(self.c_logsigma)
    )

    # Train ops
    self.vae_train_op = tf.train.AdamOptimizer(self.v_lr).minimize(
        self.vae_loss, var_list=self.vae_vars
    )
```

**APRÈS (TF2)** :
```python
class M3GAN_TF2(keras.Model):
    def __init__(self, **kwargs):
        super().__init__()

        # Optimizers
        self.vae_optimizer = keras.optimizers.Adam(learning_rate=0.0001)
        self.gen_optimizer = keras.optimizers.Adam(learning_rate=0.0001)
        self.disc_optimizer = keras.optimizers.Adam(learning_rate=0.0001)

        # Loss trackers (pour metrics)
        self.vae_loss_tracker = keras.metrics.Mean(name='vae_loss')
        self.gen_loss_tracker = keras.metrics.Mean(name='gen_loss')

    def compute_vae_loss(self, x, reconstructed, mu, log_sigma):
        """
        Calcul des losses VAE
        """
        # Reconstruction loss
        recon_loss = tf.reduce_mean(
            tf.keras.losses.mse(x, reconstructed)
        )

        # KL divergence
        kl_loss = -0.5 * tf.reduce_mean(
            1 + log_sigma - tf.square(mu) - tf.exp(log_sigma)
        )

        # Loss totale
        vae_loss = self.alpha_re * recon_loss + self.alpha_kl * kl_loss

        return vae_loss, recon_loss, kl_loss

    @tf.function
    def train_vae_step(self, c_data, d_data):
        """
        Une étape d'entraînement VAE
        """
        with tf.GradientTape() as tape:
            # Forward pass
            c_recon, c_mu, c_log_sigma = self.c_vae(c_data, training=True)
            d_recon, d_mu, d_log_sigma = self.d_vae(d_data, training=True)

            # Compute losses
            c_vae_loss, c_recon, c_kl = self.compute_vae_loss(
                c_data, c_recon, c_mu, c_log_sigma
            )
            d_vae_loss, d_recon, d_kl = self.compute_vae_loss(
                d_data, d_recon, d_mu, d_log_sigma
            )

            # Matching loss
            matching_loss = tf.reduce_mean(
                tf.square(c_mu - d_mu)
            )

            # Contrastive loss
            contrastive_loss = nt_xent_loss(c_mu, d_mu)

            # Total loss
            total_loss = c_vae_loss + d_vae_loss + \
                         self.alpha_mt * matching_loss + \
                         self.alpha_ct * contrastive_loss

        # Compute gradients
        gradients = tape.gradient(
            total_loss,
            self.c_vae.trainable_variables + self.d_vae.trainable_variables
        )

        # Apply gradients
        self.vae_optimizer.apply_gradients(
            zip(gradients,
                self.c_vae.trainable_variables + self.d_vae.trainable_variables)
        )

        # Update metrics
        self.vae_loss_tracker.update_state(total_loss)

        return {
            'vae_loss': total_loss,
            'c_recon': c_recon,
            'd_recon': d_recon,
            'matching': matching_loss
        }
```

### 5.3 Migrer contrastive loss

**AVANT (TF1)** - `Contrastivelosslayer.py` :
```python
import torch  # ⚠️ Mélange TensorFlow + PyTorch !

def nt_xent_loss(z1, z2):
    # Utilise PyTorch...
    z1_torch = torch.from_numpy(z1)
    # ...
```

**APRÈS (TF2)** :
```python
import tensorflow as tf

@tf.function
def nt_xent_loss_tf2(z1, z2, temperature=0.5):
    """
    NT-Xent loss (SimCLR) en TensorFlow pur
    """
    batch_size = tf.shape(z1)[0]

    # Normalize embeddings
    z1 = tf.math.l2_normalize(z1, axis=1)
    z2 = tf.math.l2_normalize(z2, axis=1)

    # Concatenate
    z = tf.concat([z1, z2], axis=0)  # Shape: (2*batch_size, dim)

    # Compute similarity matrix
    sim_matrix = tf.matmul(z, z, transpose_b=True) / temperature

    # Mask diagonal
    mask = tf.eye(2 * batch_size, dtype=tf.bool)
    sim_matrix = tf.where(mask, -1e9, sim_matrix)

    # Labels: positives are at distance batch_size
    labels = tf.concat([
        tf.range(batch_size, 2 * batch_size),
        tf.range(0, batch_size)
    ], axis=0)

    # Cross-entropy loss
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=labels, logits=sim_matrix
    )

    return tf.reduce_mean(loss)
```

---

## 💾 ÉTAPE 6 : REMPLACER SAVER PAR CHECKPOINT

### 6.1 Migrer sauvegardes

**AVANT (TF1)** - `m3gan.py` :
```python
def build(self):
    self.build_tf_graph()
    self.build_loss()
    self.saver = tf.train.Saver()

def save(self, global_id, model_name=None, checkpoint_dir=None):
    self.saver.save(self.sess, os.path.join(checkpoint_dir, model_name),
                    global_step=global_id)

def load(self, model_name=None, checkpoint_dir=None):
    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
    self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
```

**APRÈS (TF2)** :
```python
class M3GAN_TF2(keras.Model):
    def __init__(self, **kwargs):
        super().__init__()

        # Créer checkpoint
        self.checkpoint = tf.train.Checkpoint(
            c_vae=self.c_vae,
            d_vae=self.d_vae,
            c_gan=self.c_gan,
            d_gan=self.d_gan,
            vae_optimizer=self.vae_optimizer,
            gen_optimizer=self.gen_optimizer,
            disc_optimizer=self.disc_optimizer,
            epoch=tf.Variable(0, dtype=tf.int64)
        )

        self.checkpoint_manager = tf.train.CheckpointManager(
            self.checkpoint,
            directory='./checkpoints',
            max_to_keep=3
        )

    def save_checkpoint(self, epoch):
        """Sauvegarder checkpoint"""
        self.checkpoint.epoch.assign(epoch)
        save_path = self.checkpoint_manager.save()
        print(f"✅ Checkpoint sauvegardé : {save_path}")
        return save_path

    def load_checkpoint(self, checkpoint_path=None):
        """Charger checkpoint"""
        if checkpoint_path:
            self.checkpoint.restore(checkpoint_path)
        else:
            # Charger dernier checkpoint
            latest = tf.train.latest_checkpoint('./checkpoints')
            if latest:
                self.checkpoint.restore(latest)
                print(f"✅ Checkpoint chargé : {latest}")
                return self.checkpoint.epoch.numpy()
        return 0
```

### 6.2 Alternative : Keras Model.save()

Pour sauvegardes complètes (poids + architecture) :

```python
# Sauvegarder (format SavedModel)
model.save('saved_model/ehrMGAN_epoch_100')

# Ou format HDF5 (plus compact)
model.save('checkpoints/model_epoch_100.h5')

# Charger
model = keras.models.load_model('saved_model/ehrMGAN_epoch_100')
```

---

## ✅ ÉTAPE 7 : TESTS ET VALIDATION

### 7.1 Créer suite de tests

Créer `tests/test_migration_tf2.py` :

```python
import tensorflow as tf
import numpy as np
import pytest

class TestMigrationTF2:
    """Tests de validation de la migration TF2"""

    def test_tf2_version(self):
        """Vérifier version TF2"""
        assert tf.__version__.startswith('2.'), f"TF2 requis, version: {tf.__version__}"

    def test_vae_forward_pass(self):
        """Tester forward pass VAE"""
        from migration_tf2.networks_tf2 import C_VAE_TF2

        vae = C_VAE_TF2(batch_size=32, time_steps=24, dim=7, z_dim=25)

        # Données dummy
        x = tf.random.normal((32, 24, 7))

        # Forward pass
        reconstructed, mu, log_sigma = vae(x, training=False)

        # Assertions
        assert reconstructed.shape == x.shape
        assert mu.shape == (32, 25)
        assert log_sigma.shape == (32, 25)

        print("✅ VAE forward pass OK")

    def test_bilateral_lstm(self):
        """Tester Bilateral LSTM Cell"""
        from migration_tf2.Bilateral_lstm_tf2 import BilateralLSTMCell

        cell = BilateralLSTMCell(num_units=128)

        # Input dummy
        inputs = tf.random.normal((32, 10))
        states = [tf.zeros((32, 128)), tf.zeros((32, 128))]

        # Forward
        output, new_states = cell(inputs, states)

        # Assertions
        assert output.shape == (32, 128)
        assert len(new_states) == 2

        print("✅ Bilateral LSTM OK")

    def test_train_step(self):
        """Tester une étape d'entraînement"""
        from migration_tf2.m3gan_tf2 import M3GAN_TF2

        model = M3GAN_TF2(batch_size=32, time_steps=24)

        # Données dummy
        c_data = tf.random.normal((32, 24, 7))
        d_data = tf.random.uniform((32, 24, 3), 0, 1)

        # Train step
        losses = model.train_vae_step(c_data, d_data)

        # Vérifier losses finies (pas NaN/Inf)
        for key, val in losses.items():
            assert tf.math.is_finite(val), f"{key} loss is not finite"

        print("✅ Train step OK")

    def test_checkpoint_save_load(self):
        """Tester save/load checkpoint"""
        from migration_tf2.m3gan_tf2 import M3GAN_TF2

        model = M3GAN_TF2(batch_size=32, time_steps=24)

        # Sauvegarder
        save_path = model.save_checkpoint(epoch=10)
        assert os.path.exists(save_path + '.index')

        # Charger
        loaded_epoch = model.load_checkpoint(save_path)
        assert loaded_epoch == 10

        print("✅ Checkpoint save/load OK")

    def test_numerical_equivalence(self):
        """
        Tester équivalence numérique TF1 vs TF2
        """
        # Charger modèle TF1 (si checkpoint existe)
        # Charger modèle TF2 avec mêmes poids
        # Comparer outputs sur mêmes inputs

        # TODO: Implémenter quand migration complète
        pytest.skip("Nécessite modèles TF1 et TF2 complets")

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
```

### 7.2 Lancer tests

```bash
# Installer pytest
pip install pytest pytest-cov

# Lancer tests
pytest tests/test_migration_tf2.py -v

# Avec coverage
pytest tests/test_migration_tf2.py --cov=migration_tf2 --cov-report=html
```

### 7.3 Validation numérique

**Créer script de comparaison** `tests/compare_tf1_tf2.py` :

```python
"""
Compare outputs TF1 vs TF2 pour valider équivalence
"""

import numpy as np
import pickle

# Charger données test
with open('data/real/eicu/vital_sign_24hrs.pkl', 'rb') as f:
    test_data = pickle.load(f)[:10]  # 10 patients

# Modèle TF1 (environnement séparé)
print("Loading TF1 model...")
# ... (nécessite environnement TF1)

# Modèle TF2
print("Loading TF2 model...")
from migration_tf2.m3gan_tf2 import M3GAN_TF2
model_tf2 = M3GAN_TF2(...)
model_tf2.load_checkpoint('checkpoints/tf2_model')

# Générer avec TF2
output_tf2 = model_tf2.generate_data(num_sample=10)

# Comparer statistiques
print("\nComparaison statistique:")
print(f"Mean TF1: {output_tf1.mean():.4f}, TF2: {output_tf2.mean():.4f}")
print(f"Std TF1: {output_tf1.std():.4f}, TF2: {output_tf2.std():.4f}")

# Tolérance acceptable : ~1e-5 pour float32
diff = np.abs(output_tf1 - output_tf2).max()
print(f"Max absolute difference: {diff:.6f}")

if diff < 1e-4:
    print("✅ VALIDATION OK - Équivalence numérique confirmée")
else:
    print("⚠️  ATTENTION - Différences significatives détectées")
```

---

## 📋 CHECKLIST FINALE

### Phase 1 : Migration automatique
- [ ] Installer TensorFlow 2.10+
- [ ] Exécuter `tf_upgrade_v2` sur tout le code
- [ ] Lire `migration_report.txt` et noter warnings
- [ ] Corriger erreurs automatiques détectées

### Phase 2 : Imports et dépendances
- [ ] Remplacer `tensorflow.contrib.*` par équivalents TF2
- [ ] Migrer `l2_regularizer` vers `keras.regularizers.l2`
- [ ] Vérifier tous les imports (pas de `tf.compat.v1`)

### Phase 3 : Architecture modèle
- [ ] Supprimer `tf.Session()` de `main_train.py`
- [ ] Convertir `m3gan` en `keras.Model`
- [ ] Migrer `C_VAE_NET` → `C_VAE_TF2` (keras layers)
- [ ] Migrer `D_VAE_NET` → `D_VAE_TF2`
- [ ] Migrer `C_GAN_NET` → `C_GAN_TF2`
- [ ] Migrer `D_GAN_NET` → `D_GAN_TF2`
- [ ] Refactoriser `Bilateral_LSTM_cell` en `keras.layers.Layer`

### Phase 4 : Training loop
- [ ] Remplacer `sess.run()` par `tf.GradientTape()`
- [ ] Créer `train_vae_step()` avec `@tf.function`
- [ ] Créer `train_gan_step()` avec `@tf.function`
- [ ] Migrer tous les optimizers vers `keras.optimizers`
- [ ] Implémenter `train()` en eager mode

### Phase 5 : Losses
- [ ] Migrer reconstruction loss
- [ ] Migrer KL divergence loss
- [ ] Migrer matching loss
- [ ] Convertir contrastive loss (PyTorch → TF2)
- [ ] Migrer adversarial loss
- [ ] Migrer feature matching loss

### Phase 6 : Checkpoints
- [ ] Remplacer `tf.train.Saver` par `tf.train.Checkpoint`
- [ ] Créer `CheckpointManager`
- [ ] Tester save/load
- [ ] Vérifier compatibilité backwards (charger anciens checkpoints TF1)

### Phase 7 : Tests
- [ ] Créer suite de tests unitaires
- [ ] Test forward pass VAE
- [ ] Test forward pass GAN
- [ ] Test train step
- [ ] Test checkpoint save/load
- [ ] Validation numérique TF1 vs TF2

### Phase 8 : Documentation
- [ ] Mettre à jour README avec instructions TF2
- [ ] Documenter changements d'API
- [ ] Créer guide de migration pour utilisateurs
- [ ] Mettre à jour requirements.txt

### Phase 9 : Optimisation (optionnel)
- [ ] Ajouter `@tf.function` sur fonctions critiques
- [ ] Activer mixed precision (`tf.keras.mixed_precision`)
- [ ] Profiling avec TensorBoard
- [ ] Distribution multi-GPU si nécessaire

### Phase 10 : Déploiement
- [ ] Tester sur Google Colab avec TF2
- [ ] Créer notebook Colab mis à jour
- [ ] Dockeriser avec TF2
- [ ] CI/CD avec tests TF2

---

## 🚨 PIÈGES COURANTS

### Piège 1 : Shapes dynamiques

**Problème** :
```python
# TF1 : OK
batch_size = tf.placeholder(tf.int32, [])

# TF2 : ❌ Erreur avec @tf.function
@tf.function
def forward(x):
    batch_size = x.shape[0]  # ❌ Peut être None
    return tf.zeros((batch_size, 10))
```

**Solution** :
```python
@tf.function
def forward(x):
    batch_size = tf.shape(x)[0]  # ✅ Utiliser tf.shape() pour dynamique
    return tf.zeros((batch_size, 10))
```

### Piège 2 : Variables dans loops

**Problème** :
```python
@tf.function
def train_step(data):
    for batch in data:  # ❌ Loop Python non compatible
        loss = compute_loss(batch)
```

**Solution** :
```python
@tf.function
def train_step(data):
    for batch in tf.data.Dataset.from_tensors(data):  # ✅ tf.data
        loss = compute_loss(batch)
```

### Piège 3 : NumPy operations

**Problème** :
```python
@tf.function
def process(x):
    return np.mean(x)  # ❌ NumPy pas compatible @tf.function
```

**Solution** :
```python
@tf.function
def process(x):
    return tf.reduce_mean(x)  # ✅ TensorFlow ops
```

---

## 📚 RESSOURCES

### Documentation officielle
- [TensorFlow 2 Guide](https://www.tensorflow.org/guide)
- [Migration Guide TF1→TF2](https://www.tensorflow.org/guide/migrate)
- [Keras API](https://keras.io/api/)
- [tf_upgrade_v2](https://www.tensorflow.org/guide/upgrade)

### Exemples de migration
- [TF2 VAE Example](https://www.tensorflow.org/tutorials/generative/cvae)
- [TF2 GAN Example](https://www.tensorflow.org/tutorials/generative/dcgan)
- [Custom Training Loops](https://www.tensorflow.org/guide/keras/writing_a_training_loop_from_scratch)

### Outils
- [TensorBoard](https://www.tensorflow.org/tensorboard) - Debugging
- [tf.function guide](https://www.tensorflow.org/guide/function) - Performance
- [Mixed Precision](https://www.tensorflow.org/guide/mixed_precision) - Speedup

---

## 🎯 PROCHAINES ÉTAPES

1. **Commencer par les tests** : Créer `test_migration_tf2.py` avant de migrer
2. **Migrer par couches** : Imports → Architecture → Training → Losses
3. **Valider continuellement** : Lancer tests après chaque changement
4. **Documenter** : Noter tous les changements d'API pour les utilisateurs

**Temps estimé total** : 30-40 heures pour migration complète + tests

**Bonne migration ! 🚀**
