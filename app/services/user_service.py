import hashlib
import json
import logging
import re
from datetime import datetime, timedelta

logging.getLogger().setLevel(logging.INFO)


def get_hashed(txt: str):
    """
    Returns the hashed value of the text input.

    Args:
        txt (str): The string to be hashed.

    Returns:
        str: The hexadecimal representation of the SHA-256 hash of the input text.
    """
    return hashlib.sha256(txt.encode()).hexdigest()


def check_user_type(user_id: str, db_conn=None, client=None):
    """
    Returns the type of user - internal/external/tiger.

    Args:
        user_id (str): The user ID whose user type needs to be checked.
        db_conn (connection, optional): The database connection. Defaults to None.
        client (client, optional): Specifies the database core client. Defaults to None.

    Returns:
        str: The type of the user: 'internal', 'external', or 'tiger'.
    """
    if db_conn is None:
        raise Exception("No Database Connection Provided")

    tiger_email = re.compile("[a-zA-Z0-9._-]+@tigeranalytics.com")

    if re.search(tiger_email, user_id):
        return "tiger"
    else:
        query = (
            f"""select user_type from user_credentials where user_id = ('{user_id}')"""
        )
        return client.execute_query(connection=db_conn, query=query, return_value=True)


def check_reqs_limit(user_id: str, db_conn=None, client=None, user_type=None):
    """
    Checks if the user has reached the request limit.

    Args:
        user_id (str): The user ID whose request limit needs to be checked.
        db_conn (connection, optional): The database connection. Defaults to None.
        client (client, optional): The database core client. Defaults to None.

    Returns:
        int: The remaining number of requests allowed if the user hasn't reached the limit,
             None if there's an error, requests exceeds the limit or no database connection provided.
    """
    try:
        if db_conn is None:
            raise Exception("No Database Connection Provided")
        if user_type == "external":
            limit = 50
            query = f"""select total_requests from user_credentials where user_id = ('{user_id}')"""
            reqs = client.execute_query(connection=db_conn, query=query, return_value=True)
            if reqs < limit:
                return limit - reqs
        else:
            # internal/ admins have always 1000 limit. lucky fellows.
            return 1000
    except Exception as e:
        logging.info("Exception Occured In validating the user request limit.")


def get_user_details_from_token(token: str):
    # TODO: Decrypt the token ...
    # TODO: Get name, oid(user_id), and unique_name(email), login_time and store into user_config table
    pass


def create_user_cred(
    user_id: str,
    pwd: str,
    usertype: str,
    domain: str,
    exp_date=None,
    db_conn=None,
    client=None,
):
    """
    Creates user credentials in the database.

    Args:
        user_id (str): The user ID for the new user.
        pwd (str): The password for the new user.
        usertype (str): The type of the user: 'internal', 'external', or 'tiger'.
        domain (str): The domain/s user will have access to.
        exp_date (str, optional): The expiry date of the user credentials in the format 'YYYY-MM-DD'.
                                  Defaults to None.
        db_conn (connection, optional): The database connection. Defaults to None.
        client (client, optional): The database core client. Defaults to None.

    Returns:
        str: A message indicating the success or failure of the user credentials creation.
    """
    try:
        if db_conn is None:
            raise Exception("No Database Connection Provided")

        if exp_date is None:
            exp_date = datetime.utcnow() + timedelta(days=7)
        else:
            exp_date = datetime.strptime(exp_date, "%Y-%m-%d").date()
            if exp_date <= datetime.utcnow().date():
                raise Exception("Expiry date should be in future !!")

        pwd = get_hashed(pwd)

        try:
            logging.info("--Executing Query--")
            query = f"""insert into user_credentials(user_id, password, user_type, domain, expiry_date)
                     values ('{user_id}', '{pwd}', '{usertype}', '{domain}', '{exp_date}')"""
            client.execute_query(connection=db_conn, query=query)
            logging.info("--Query Executed--")

        except:
            raise Exception("Query Execution Failed | User Already Exists")

        return "User Credentials Created"

    except Exception as e:
        return e.args[0]


def validate_user_cred(user_id: str, pwd: str, db_conn=None, client=None):
    """
    Validates user credentials and provides additional information based on the user type.

    Args:
        user_id (str): The user ID needs to be verified.
        pwd (str): The password associated with the user.
        db_conn (connection, optional): The database connection. Defaults to None.
        client (client, optional): The database core client. Defaults to None.

    Returns:
        Union[str, dict]: If the user is successfully validated, a JSON string containing
                        message, user_type, and requests_remaining.
    """
    try:
        if db_conn is None:
            raise Exception("No Database Connection Provided")

        pwd = get_hashed(pwd)
        rem = 0
        lastlog = datetime.now()
        today = datetime.now()
        logging.info(f"Validating the user : {user_id}")
        query = f"""select expiry_date from user_credentials where user_id = ('{user_id}') and password = ('{pwd}')"""
        expiry = client.execute_query(connection=db_conn, query=query, return_value=True)
        logging.info(f"expiry : {expiry}")
        if not expiry:
            # either user is not found or expiry is null
            logging.info(f"user not found.")
            return f"User '{user_id}' not found. Please check with admin."
        else:
            if today >= expiry:
                # user is found and validity expired.
                logging.info(f"user validity expired. Not dead though.")
                return f"Validity Expired for user : '{user_id}'"
            else:
                # User is found and vaidity not expired
                logging.info(f"User '{user_id}' exist.")
            
                user_type = check_user_type(user_id=user_id, db_conn=db_conn, client=client)
                logging.info(f"User '{user_id}' is an {user_type} user.")

                requests_remaining = check_reqs_limit(user_id=user_id, db_conn=db_conn, client=client, user_type=user_type)
                logging.info(f"Requests remaining: {requests_remaining}")
                if requests_remaining <= 0 :
                    # poor fellow wasted asking too many questions.
                    logging.info(f"user exausted login limits.")
                    return "User '{user_id}' have exhausted the login limit !!"
                else:
                    query = f"""update user_credentials set last_login = ('{lastlog}') where user_id = ('{user_id}')"""
                    client.execute_query(connection=db_conn, query=query)
                    logging.info("user login log updated")

                    response = {
                    "message": f"User Verified! Validity expires in {(expiry-today).days} days",
                    "user_type": user_type,
                    "requests_remaining": requests_remaining
                               }
                    return json.dumps(response)
    except Exception as e:
        logging.ERROR(f"Exception occured while validating the user.\n\nError : \n\n{str(e)}")


def get_user_domains(user_id: str, db_conn=None, client=None):
    """
    Retrieves the domains associated with a user from the database.

    Args:
        user_id (str): The user ID for which domain/s need to be retrieved.
        db_conn (connection, optional): The database connection. Defaults to None.
        client (client, optional): The database core client. Defaults to None.

    Returns:
        list: A list of domain strings associated with the user, or None if there's an error or no domains found.
    """
    try:
        if db_conn is None:
            raise Exception("No Database Connection Provided")

        query = f"""select domain from user_credentials where user_id = ('{user_id}')"""
        # logging.info("--Executing Query--")
        domains = client.execute_query(
            connection=db_conn, query=query, return_value=True
        )
        # logging.info("--Query Executed--")

        domains = [i.strip() for i in domains.split(",")]

        return domains

    except:
        return None
