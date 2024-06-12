import pandas as pd
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt


def load_csv():
  df = pd.read_excel('table.xlsx')
  df.dropna(subset=['ds_instruction'], inplace=True)

  df['time'] = np.random.randint(1, 6, df.shape[0])

  null_summary = df.isnull().sum()
  print(df.head())
  """ print("Summary of null values in each column:")
  print(null_summary) """

  return df


class PrecedenceGraph:
  def __init__(self, initial_df):
    self.initial_df = initial_df
    self.solutions = []

  def __len__(self):
    return len(self.solutions)

  def from_tb_balancing_graph(self, number_of_posts=12):
    if ('id_build_schema_wi_log' in self.initial_df.columns):
      # randomize the order
      shuffled_df = self.initial_df.sample(frac=1).reset_index(drop=True)

      # divide the shuffled df to add them to the blocks
      parts = np.array_split(shuffled_df, number_of_posts)

      main_graph = nx.DiGraph()

      # Add nodes numbered from 1 to 12 to the main graph
      for i in range(1, number_of_posts+1):
        main_graph.add_node(i)

      # Function to create a subgraph from a DataFrame part
      def create_subgraph(df_part):
        G = nx.DiGraph()
        for i in range(len(df_part) - 1):
          source = df_part.iloc[i]['id_build_schema_wi_log']
          target = df_part.iloc[i + 1]['id_build_schema_wi_log']
          G.add_edge(source, target)
        return G

      # create subgraphs for each post
      subgraphs = {}
      for i in range(number_of_posts):
        subgraph = create_subgraph(parts[i])
        subgraphs[i + 1] = subgraph
        main_graph.add_node(i + 1, subgraph=subgraph)

      # append solution
      self.solutions.append(
          {"id": f"graph-{self.__len__() + 1}", "graph": main_graph})

  def generate_n_balacing_graphs(self, n):
    for i in range(n):
      self.from_tb_balancing_graph()

    """ for solution in self.solutions:
      print(solution) """

  def is_solution_valid(self, id):
    # find solution
    solution = next(
        (item for item in self.solutions if item['id'] == id), None)

    activities_in_correct_post = []
    activities_in_wrong_post = []

    # validate post subgraph
    def check_subgraph(graph_of_post_activities, current_post):
      post_activities_df = pd.DataFrame(columns=self.initial_df.columns)

      for node in graph_of_post_activities.nodes(data=True):
        matched_activity = self.initial_df[self.initial_df['id_build_schema_wi_log'] == node[0]]
        if not matched_activity.empty:
          post_activities_df = post_activities_df._append(
              matched_activity, ignore_index=True)

      for index, activity in post_activities_df.iterrows():
        if (activity["Unnamed: 3"] and isinstance(activity["Unnamed: 3"], str)):
          required_activity_post = int(activity["Unnamed: 3"][-2:])
          if (required_activity_post != current_post):
            activities_in_wrong_post.append(
                {"activity_id": activity["id_build_schema_wi_log"], "post": current_post, "required_activity_post": required_activity_post, "post_id": index})
          else:
            activities_in_correct_post.append(
                {"activity_id": activity["id_build_schema_wi_log"], "post": current_post, "required_activity_post": required_activity_post, "post_id": index})

    # iterate throw each post
    for current_post, graph_of_post_activities in solution["graph"].nodes(data=True):
      graph_of_post_activities = graph_of_post_activities["subgraph"]
      check_subgraph(graph_of_post_activities, current_post)

    # show results
    print("\n")
    print("activities in correct post: ", len(
        activities_in_correct_post), "\n")
    print("activities in wrong post: ", len(activities_in_wrong_post), "\n")

    if (len(activities_in_wrong_post) == 0):
      return True
    else:
      return False

  def get_solution_time(self, id):
    # find solution
    solution = next(
        (item for item in self.solutions if item['id'] == id), None)
    post_times = []

    # get time of post subgraph
    def get_subgraph_time(graph_of_post_activities):
      time = 0
      for node in graph_of_post_activities.nodes(data=True):
        matched_activity = self.initial_df[self.initial_df['id_build_schema_wi_log'] == node[0]]
        time += int(matched_activity["time"].iloc[0])
      return time

    # iterate throw each post
    for current_post, graph_of_post_activities in solution["graph"].nodes(data=True):
      graph_of_post_activities = graph_of_post_activities["subgraph"]
      post_times.append(get_subgraph_time(graph_of_post_activities))

    # show results
    for index, item in enumerate(post_times):
      print("post ", index+1, " with ", item, " seconds")


pg = PrecedenceGraph(initial_df=load_csv())

pg.generate_n_balacing_graphs(n=10)

pg.is_solution_valid(id="graph-1")

pg.get_solution_time(id="graph-1")
