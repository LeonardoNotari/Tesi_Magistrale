Nella repository sono presenti i seguenti script python:
- 'decoder.py': il modello per decodificare le rappresentazioni di ULIP2 in point cloud.
- 'train.py': Il codice per il training che utilizza la Chamfer Loss e il subsampling per ridurre la dimensione nella gpu.
- 'dataloader.py': script per caricare il dataset
- 'pc_trasformation.py': a partire da una point-cloud e da una coppia di testi che indicano un polo positivo e negativo di una determinata caratteristica, calcola l'embedding della point cloud con un modello ULIP2 e la direzione data dai due testi nello spazio latente (se sono entrambi presenti non carica il checkpoint di ULIP2 che è un'operazione che richiede alcuni minuti), sposta quindi l'embedding linearmente secondo un valore 'ALPHA' lungo la direzione e lo normalizza di nuovo nell'ipersfera. Se è stato scelto un valore di 'ADAPT_STEPS'>0 viene eseguito un addestramento del checkpoint del decoder sull'istanza da modificare, infine genera la point cloud corrispondente al nuovo punto dello spazio latente (salva il risultato con il nome il valore di 'ALPHA', i due testi e il numero di 'ADAPT_STEP' nella cartella dove è stata trovata la point cloud originale)

Nelle tre cartelle 'trasform_input_face', 'trasform_input_donkey' e 'trasform_input_spot' sono presenti alcuni
esempi di esecuzione del codice del file 'pc_trasformation.py' a partire da diverse point cloud di partenza per valori diversi di 'ADAPT-STEP' e 'ALPHA'.

Sì, è comprensibile e abbastanza chiaro. Ti suggerisco però una versione leggermente rivista per corregere piccoli dettagli grammaticali, uniformare lo stile e renderlo più leggibile in un README tecnico.

---

# README

Nella repository sono presenti i seguenti script Python:

* `decoder.py`: modello utilizzato per decodificare le rappresentazioni ULIP2 in point cloud.

* `train.py`: codice per il training del decoder. Utilizza la Chamfer Loss e una procedura di subsampling per ridurre l’utilizzo di memoria GPU.

* `dataloader.py`: script per il caricamento del dataset.

* `pc_transformation.py`: a partire da una point cloud e da una coppia di testi che definiscono un polo positivo e uno negativo di una determinata caratteristica:

  1. calcola l’embedding della point cloud tramite un modello ULIP2;
  2. calcola la direzione nello spazio latente definita dai due testi;
  3. se gli embedding testuali sono già presenti, evita di ricaricare il checkpoint ULIP2 (operazione che richiede alcuni minuti);
  4. sposta linearmente l’embedding lungo tale direzione tramite il parametro `ALPHA`;
  5. normalizza nuovamente il punto sull’ipersfera latente;
  6. se è stato scelto un valore `ADAPT_STEPS > 0`, viene inoltre eseguito un fine-tuning del checkpoint del decoder sulla specifica istanza da modificare;
  7. viene generata la point cloud corrispondente al nuovo punto nello spazio latente;
  8. il risultato viene salvato nella stessa cartella della point cloud originale, con un nome che include:
      * valore di `ALPHA`;
      * testi positivo e negativo;
      * numero di `ADAPT_STEPS`.

Nelle cartelle:

* `transform_input_face`
* `transform_input_donkey`
* `transform_input_spot`

sono presenti alcuni esempi di esecuzione dello script `pc_transformation.py`, ottenuti a partire da diverse point cloud iniziali e con differenti valori di `ADAPT_STEPS` e `ALPHA`.
