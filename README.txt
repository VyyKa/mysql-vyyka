You can edit password for MySQL server by edit parameter of: 'MYSQL_ROOT_PASSWORD' on docker-compose.yaml file
if not, 'example' is default password

Create logs file, and grant permission for other user (Logtash/Docker) be able to read log 
Run command:
chmod +x permission.sh

sudo ./permission.sh

docker-compose up -d --build

Now, exec to MySQL container and test some basic command such as: 
show databases;
use mysql;
show tables;
...
or try to create user and grant permission
and then get back to host to check /logs/mysql.log to see if logs have been written.
if not
First: 
check VARIBLES 'general-log%' by command: 
SHOW VARIABLES LIKE 'general_log%';

It expect to be like this:
| general_log      | ON  |
| general_log_file | /var/log/mysql/mysql.log |

If general_log is OFF
Then exec back to mysql container and run command:
SET GLOBAL general_log = 'ON';

Check log of AI:
docker-compose logs -f ai-app
