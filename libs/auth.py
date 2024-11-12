from flask_login import LoginManager

login_manager = LoginManager()
def login_init(app):
    login_manager.init_app(app)
