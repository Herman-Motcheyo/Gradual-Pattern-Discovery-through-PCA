
class CustomError(Exception):
    
    def __init__(self, df_name, message="Erreur de traitement"):
        self.df_name = df_name
        self.message = self.message 
        super().__init__(self.message)

    def __str__(self):
        return f"{self.message} : {self.df_name}"

    def log_error(self, log_file = 'error_log.txt'):

        with open(log_file, "a") as f:
            f.write(f"Erreur dans {self.dataframe_name} : {self.message}\n")
