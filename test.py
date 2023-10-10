from multi_funcs import *
from funcs import *
from os import listdir
from os.path import isfile, join
import numpy as np
import pandas as pd
import memory_profiler
import sys
sys.setrecursionlimit(4000)

# Python program to find strongly connected components in a given
# directed graph using Tarjan's algorithm (single DFS)
#Complexity : O(V+E)

from collections import defaultdict

# This class represents an directed graph
# using adjacency list representation


class Graph:

	def __init__(self, vertices):
		# No. of vertices
		self.V = vertices

		# default dictionary to store graph
		self.graph = defaultdict(list)

		self.Time = 0
		self.Cycle=0
		self.cycles = []

	# function to add an edge to graph
	def addEdge(self, u, v):
		self.graph[u].append(v)

	'''A recursive function that find finds and prints strongly connected
	components using DFS traversal
	u --> The vertex to be visited next
	disc[] --> Stores discovery times of visited vertices
	low[] -- >> earliest visited vertex (the vertex with minimum
				discovery time) that can be reached from subtree
				rooted with current vertex
	st -- >> To store all the connected ancestors (could be part
		of SCC)
	stackMember[] --> bit/index array for faster check whether
				a node is in stack
	'''

	def SCCUtil(self, u, low, disc, stackMember, st):

		# Initialize discovery time and low value
		disc[u] = self.Time
		low[u] = self.Time
		self.Time += 1
		stackMember[u] = True
		st.append(u)

		# Go through all vertices adjacent to this
		for v in self.graph[u]:

			# If v is not visited yet, then recur for it
			if disc[v] == -1:

				self.SCCUtil(v, low, disc, stackMember, st)

				# Check if the subtree rooted with v has a connection to
				# one of the ancestors of u
				# Case 1 (per above discussion on Disc and Low value)
				low[u] = min(low[u], low[v])

			elif stackMember[v] == True:

				'''Update low value of 'u' only if 'v' is still in stack
				(i.e. it's a back edge, not cross edge).
				Case 2 (per above discussion on Disc and Low value) '''
				low[u] = min(low[u], disc[v])

		# head node found, pop the stack and print an SCC
		w = -1 # To store stack extracted vertices
		if low[u] == disc[u]:
			j = 0
			list1 = []
			while w != u:
				w = st.pop()
				#print(w, end=" ")
				j+=1
				list1.append(w)
				stackMember[w] = False
			if j > 1:
				self.Cycle+=1
				self.cycles.append(list1)
			#print()

	# The function to do DFS traversal.
	# It uses recursive SCCUtil()

	def SCC(self):

		# Mark all the vertices as not visited
		# and Initialize parent and visited,
		# and ap(articulation point) arrays
		disc = [-1] * (self.V)
		low = [-1] * (self.V)
		stackMember = [False] * (self.V)
		st = []

		# Call the recursive helper function
		# to find articulation points
		# in DFS tree rooted with vertex 'i'
		for i in range(self.V):
			if disc[i] == -1:
				self.SCCUtil(i, low, disc, stackMember, st)
		return self.Cycle, self.cycles



df = mdf_np(4000, 60)
preferences = df.copy()
matches, _ = run_gale_shapley(df, 30)
matches
print(len(matches))
preferences = preferences[preferences['student_id'].isin(matches['student_id'])]
preferences.reset_index(inplace = True, drop = True) 
preferences['rejections'] = matches.applications
relevant = preferences[preferences['rejections'] != 0]
for i in range(1,30):
    relevant.iloc[:, i] = np.where(relevant['rejections']<i+1, -100, relevant.iloc[:, i])
relevant.set_index('student_id', inplace = True)
pointing = pd.DataFrame(relevant.iloc[:, :30].stack(level = 0)).reset_index()
pointing = pointing[pointing[0] != -100]
to_merge = matches.loc[:,[0, 'student_id']]
pointing = pointing.merge(to_merge, on = 0, how = 'left')
pointing = pointing[pointing['student_id_y'].isin(pointing['student_id_x'])]
normalizer = pd.DataFrame(pd.concat([pointing['student_id_x'], pointing['student_id_y']], axis = 0).unique())
normalizer['new_id'] = normalizer.index
pointing = pointing.merge(normalizer, left_on = 'student_id_x', right_on = 0, how = 'left')
pointing = pointing.merge(normalizer, left_on = 'student_id_y', right_on = 0, how = 'left')
pairs = pointing[['new_id_x', 'new_id_y']]
pairs
g = Graph(len(normalizer))
for i in range(len(pairs)):
    g.addEdge(pairs.iloc[i, 0], pairs.iloc[i, 1])
pairs, Cycle = g.SCC()
print(Cycle)
print('hi')