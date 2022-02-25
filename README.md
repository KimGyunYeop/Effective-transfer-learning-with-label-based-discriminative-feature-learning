### Experimnet code of "Effective transfer learning with label-based discriminative feature learning"

requirement package is writing in requirement.txt

This project requires the following data format.(.tsv)

<pre>
<code>
data  label  
dddaaatttttaaaaaa1  label1  
ddaaattaaa2  label2  
</code>
</pre>

The execution code is as follows.  
The hyperparameter may be changed in all_file_train.sh.  
Model_mode = \[Basemodel, Star_Label_AM, Star_Label_ANN\]

  bash all_file_train.sh
