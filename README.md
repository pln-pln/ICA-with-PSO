# ICA-with-PSO
This is an ongoing project launched to study the change in the performance of independent component analysis with FastICA when it is combined with particle swarm optimization. Independent component analysis is a blind source separation technique that is proved to be successful and is widely accepted. The ICA algorithm used in this project utilizes maximization of negentropy to find a demixing matrix. The idea behind the project is to use particle swarm optimization to find this demixing matrix with the help of particle swarm optimization to see whether the results are improved. 
The two elements of each of the rows of demixing matrix are assigned to be particle coordinates. The coordinates are input to the fitness function which utilizes negentropy. Upon evaluation of the fitness after each iteration, particle coordinates are updated until the best fit is reached. The demixing matrix is returned after this process is repeated for each row. Other steps of the ica algorithm and the preprocessing steps are not changed.
The combined algorithm is able to separate independent components successfully. Numerical evaluation of its success and comparison to the original ica algorithm are yet to be completed or documented.

The PSO and fitnessfunction files contain the code for this project. Other two files, ica and pso_original are the reference codes combined in this project. They were added here for demonstration.

Plot of the results:
![Screenshot (63)](https://user-images.githubusercontent.com/69345688/195174007-5f1debcc-4aa0-463a-872a-d1017c75df28.png)
