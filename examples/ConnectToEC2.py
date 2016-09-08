#!/usr/bin/env python
"""
Written by Vijay MK Namboodiri and Randall L Ung of the Stuber lab at UNC-CH

Disclaimer: Some of the structure of this code is borrowed from a similar
 script included in Thunder (http://thunder-project.org/) for the set up of
 a cluster on AWS. There is no actual or implied warranty for the use of
 this script. Users are highly encouraged to test the script for bugs.

Use: This script launches a remote instance on Amazon Web Services (AWS) EC2
 for the purpose of running SIMA. The script should work on Windows, MAC and
 Linux. This is a modified version of the Thunder script intended to install
 SIMA and its dependencies on an EC2 instance. It can also automatically
 launch an instance, analyze a custom script and terminate the instance when
 the custom analysis is done. Run python ConnectToEC2.py -h for more detailed
 help.

Installation: We recommend that users install Python through Anaconda
 (https://www.continuum.io/downloads). In addition to the standard packages
 that come with Anaconda, you will need to install termcolor, paramiko and
 colorama (for Windows) to run this script.

Setting up AWS EC2: You can either follow
 http://docs.aws.amazon.com/AWSEC2/latest/UserGuide/get-set-up-for-amazon-ec2
.html
 or the instructions below.

Go to AWS to sign up for an account. Once you have created an account, go
 to "Identity and Access Management", select "Users" on the left, and click
 "Create New Users" at the top. Follow the instructions to create a user for
 yourself. After you create a user, click "Show User Security Credentials"
 to see your access key ID and secret access key (long strings of
 characters/numbers). These are the security credentials used to log in to
 AWS. Write these down, or click download to save them to a file, and close
 the window. These will be used below to set environment variables as
 described in the section below (Local set-up).

Click the checkbox next to your user name. In the window that appears at the
 bottom of the page, select "Permissions > Attach user policy", select
 Administrator Access policy, and click "Apply Policy". This will give you
 administrator access as the current user. If you opened a brand new account,
 it may take a couple hours for Amazon to verify it, so wait before
 proceeding.

Once you have done the above set up, you need to create a key-pair for
 logging in to EC2 through SSH. To do this, go to the Key-Pair section
 on the EC2 console. Click on the region name in the upper right (it will
 probably say Oregon), and select US East (N Virginia). Then click
 "Create Key Pair" at the top. Give it any name you want, but note that
 our script has a default name set to "mykey". So if your key is indeed
 named "mykey", you will not have to specify the key name every time you
 run the script. This eases the use for running the script. You will
 download a file with an extension of .pem. The default location assumed
 by our script is in the directory containing the script. If it is stored
 elsewhere, you will have to specify the path to the file every time you
 run the script.

Make sure to set the correct permissions on this file by typing
 chmod 600 ./mykey.pem
 i.e. only owner can read and write. Without this, the key file will be
 rejected by AWS when you ssh.

Local set-up: The script reads your AWS credentials from environment
 variables. This is handled by the python module
 called boto. Set your AWS credentials as the following environment variables:
1. AWS_ACCESS_KEY_ID = <youraccessid>
2. AWS_SECRET_ACCESS_KEY = <yoursecretaccesskey>

On Windows, this can be done by following the instructions here:
https://www.microsoft.com/resources/documentation/windows/xp/all/proddocs/
en-us/sysdm_advancd_environmnt_addchange_variable.mspx?mfr=true

On Unix systems, follow http://unix.stackexchange.com/questions/21598/
how-do-i-set-a-user-environment-variable-permanently-not-session

On a Mac, follow http://stackoverflow.com/questions/7501678/
set-environment-variables-on-mac-os-x-lion

Additional steps on Windows:
1. The script uses paramiko to ssh into EC2. However, for the purposes of
 logging in and remotely accessing the terminal of the EC2 instance, we
 use Plink (Putty-link). Users must download Plink.exe and Puttygen.exe from
http://www.chiark.greenend.org.uk/~sgtatham/putty/download.html

2. Plink requires ssh private key files in a .ppk format instead of the .pem
 format that AWS requires. Users must convert the .pem key file into a .ppk
 key file: load the .pem file into Puttygen, convert it, and save it in the
 same directory as the .pem file with the same file name. So mykey.pem should
 be stored as mykey.ppk. Both of these files should be present.

One last thing to keep in mind is the type of instance that you
 want to start. The default below is r3.2xlarge, which is a memory
 optimized instance. A list of all instances can be found here:
 http://www.ec2instances.info/ The one unintuitive thing to keep
 in mind when choosing an instance is that with SIMA motion correction,
 a lot of RAM might end up getting used as of July 13, 2016.
 This issue is discussed on Github here:
 https://github.com/losonczylab/sima/issues/216
 So until this is resolved, users will want to launch an instance with RAM
 larger than the size of all files combined. This is just a rough
 approximation for the default settings of SIMA. For instance, if
 the total size of all your files is less than ~50-55GB, you can use
 r3.2xlarge, which has 62GB of RAM. If the RAM gets fully used, the
 instance throws SSH connection errors. At that point, there is no
 other option but to terminate the instance and start another one with
 more RAM.
"""

from boto import ec2
from termcolor import colored
import paramiko
from datetime import datetime
import subprocess
import time
import random
import sys
import os
import socket
from argparse import ArgumentParser, RawTextHelpFormatter

# Enable termcolor in Windows
if os.name == 'nt':
    import colorama
    colorama.init()


def print_status(msg):
    print("    [" + msg + "]")


def print_success(msg="success"):
    print("    [" + colored(msg, 'green') + "]")


def print_error(msg="failed"):
    print("    [" + colored(msg, 'red') + "]")


def print_remote_message(msg):
    print(colored(msg, 'cyan'))
    with open("analysislog.txt", "a") as myfile:
        myfile.write(msg)


def install_anaconda(ssh_client):
    """ Install Anaconda on an EC2 machine """

    # download anaconda
    print_status("Downloading Anaconda")
    ssh(ssh_client, "rm -rf ./Anaconda-2.3.0-Linux-x86_64.sh && "
        "wget https://3230d63b5fc54e62148e-c95ac804525aac4b6dba79b00b39d1d3.ssl.cf1.rackcdn.com/Anaconda-2.3.0-Linux-x86_64.sh")
    print_success()

    # setup anaconda
    print_status("Installing Anaconda")
    ssh(ssh_client, "rm -rf /root/anaconda && "
                    "bash ./Anaconda-2.3.0-Linux-x86_64.sh -b && "
                    "rm ./Anaconda-2.3.0-Linux-x86_64.sh && "
                    "echo 'export PATH=/root/anaconda/bin:$PATH' >> \
                           /root/.bash_profile &&"
                    "sed -i -e '1iexport PATH=/root/anaconda/bin:$PATH\' \
                           /root/.bashrc")
    # for non-interactive login using ssh. The path has to be at the top of
    # .bashrc. Otherwise, bashrc has the line "#
    # If not running interactively, don't do anything\ [ -z "$PS1" ] &&
    # return" which prevents any change for
    # non-interactive ssh login. So, the environmental variables will not be
    # set while sshing. This is used for the
    # action analyzeandterminate for this script.
    print_success()

    # update core libraries
    print_status("Updating Anaconda libraries")
    ssh(ssh_client, "/root/anaconda/bin/conda update --yes numpy scipy \
                     ipython scikit-image h5py &&"
                    "/root/anaconda/bin/conda install --yes jsonschema \
                     pillow seaborn scikit-learn shapely bottleneck "
                    "future &&"
                    "/root/anaconda/bin/pip install jupyter")
    print_success()


def install_sima(ssh_client):
    print_status("Installing SIMA")
    install_git(ssh_client)
    ssh(ssh_client, "/root/anaconda/bin/pip install \
                     git+git://github.com/vjlbym/sima@master")
    print_success()


def install_git(ssh_client):
    print_status("Installing Git")
    ssh(ssh_client, "sudo apt-get -y install git")
    print_success()


def install_opencv(ssh_client):
    print_status("Updating apt-get")
    ssh(ssh_client, "apt-get update")
    print_success()

    print_status("Installing gcc")
    ssh(ssh_client, "apt-get install gcc --yes")
    print_success()

    print_status("Installing opencv")
    ssh(ssh_client, "sudo apt-get install libopencv-dev python-opencv --yes")
    print_success()


def install_blaslapack(ssh_client):
    print_status("Installing blas and lapack for picos")
    ssh(ssh_client, "sudo apt-get install libblas3gf libblas-doc libblas-dev \
                     liblapack3gf liblapack-doc liblapack-dev "
                    "--yes")
    print_success()


def install_picos(ssh_client):
    print_status("Installing picos")
    install_blaslapack(ssh_client)  # Needed for picos to be installed properly
    ssh(ssh_client, "/root/anaconda/bin/pip install picos")
    print_success()


def install_MDP(ssh_client):
    print_status("Install MDP")
    ssh(ssh_client, "sudo apt-get install python-mdp --yes")
    print_success()


def setup_sign_of_life(ssh_client):
    ssh(ssh_client, "sudo echo 'MaxSessions=100' >> \
                     /etc/ssh/sshd_config")
    ssh(ssh_client, "sudo echo 'ClientAliveInterval 50' >> \
                     /etc/ssh/sshd_config")
    ssh(ssh_client, "sudo chmod 600 /etc/ssh/sshd_config && \
                     sudo service ssh restart")
    print_status("Set up instance to send sign-of-life ssh packets")


def ssh_connect(host, opts):
    ssh_client = paramiko.SSHClient()
    ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    # do not prompt new connections
    if os.name == 'posix':
        ssh_client.save_host_keys('/dev/null')  # do not save host keys
    elif os.name == 'nt':
        ssh_client.save_host_keys('NUL')  # do not save host keys on Windows
    ssh_client.connect(host,
                       username=opts.user,
                       key_filename=opts.identity_file)
    return ssh_client


def ssh(ssh_client, command, block=True, get_pty=False, display=False):
    ssh_chan = ssh_client.get_transport().open_session()
    if get_pty:
        ssh_chan.get_pty()
    ssh_chan.exec_command(command)
    if block:
        analysis_failed_flag = False
        # runanalysis is run through a dtach session. The dtach session
        # exits successfully even if the analysis actually failed. So checking
        # for exit flag from ssh would not detect the failure of the analysis.
        # Here, I am checking the input stream from the ssh connection to see
        # if the string "Analysis failed" is printed onto the instance's
        # stdout. If so, it means that the analysis actually failed even
        # though the exit flag from dtach was 0. This is caught as an
        # AnalysisError class, defined below.

        duration = 0
        timeout = 30*60  # in seconds
        temp = ''
        if display:
            start_time = time.time()
            while not ssh_chan.exit_status_ready() and duration < timeout:
                duration = time.time() - start_time
                if ssh_chan.recv_ready():
                    temp = ssh_chan.recv(1024)
                    print_remote_message(temp)
                    if "Analysis failed" in temp:
                        analysis_failed_flag = True
        if duration >= timeout:
            raise TimeoutError()
        else:
            # Wait for remote command to complete
            exit_code = ssh_chan.recv_exit_status()

            if not exit_code == 0:
                print "Exit code: " + str(exit_code)
                while ssh_chan.recv_stderr_ready():
                    print ssh_chan.recv_stderr(1024)
                raise Exception('Error in SSH command.')
            elif analysis_failed_flag:
                raise AnalysisError(
                     'Analysis failed. Check log file on S3 to see why')
    return temp


class AnalysisError(Exception):
    def __int__(self, msg):
        self.msg = msg


class TimeoutError(Exception):
    def __int__(self, msg):
        self.msg = msg


# Check whether a given EC2 instance object is in a state we consider active,
# i.e. not terminating or terminated. We count both stopping and stopped as
# active since we can restart stopped instances.
def is_active(instance):
    return instance.state in ['pending', 'running', 'stopping', 'stopped']


def get_existing_instance(conn, opts, die_on_error=True, outputflag=True):
    if outputflag:
        print "Searching for existing instance " + opts.instance_name + "..."

    reservations = conn.get_all_reservations()
    master_nodes = []
    for res in reservations:
        active = [i for i in res.instances if is_active(i)]
        for inst in active:
            group_names = [g.name for g in inst.groups]
            if opts.instance_name in group_names:
                master_nodes.append(inst)
    if any(master_nodes):
        if outputflag:
            print "Found %d instance(s)" % (len(master_nodes))
    if master_nodes != [] or not die_on_error:
        return master_nodes
    else:
        print >> sys.stderr, "ERROR: Could not find any existing instance"
        sys.exit(1)


def launch_instance(conn, opts):
    if opts.identity_file is None:
        print >> sys.stderr, "ERROR: Must provide an identity file \
                             (-i) for ssh connections."
        sys.exit(1)
    if opts.key_pair is None:
        print >> sys.stderr, "ERROR: Must provide a key pair name (-k) \
                              to use on instances."
        sys.exit(1)

    user_data_content = None

    print "Setting up security groups..."
    master_group = get_or_make_group(conn, opts.instance_name, opts.vpc_id)
    authorized_address = opts.authorized_address
    if not master_group.rules:  # Group was created just now
        master_group.authorize(src_group=master_group)
        master_group.authorize('tcp', 22, 22, authorized_address)
        master_group.authorize('tcp', 8888, 8888, authorized_address)
        master_group.authorize('tcp', 8080, 8081, authorized_address)
        master_group.authorize('tcp', 18080, 18080, authorized_address)
        master_group.authorize('tcp', 19999, 19999, authorized_address)
        master_group.authorize('tcp', 50030, 50030, authorized_address)
        master_group.authorize('tcp', 50070, 50070, authorized_address)
        master_group.authorize('tcp', 60070, 60070, authorized_address)
        master_group.authorize('tcp', 4040, 4045, authorized_address)

    # Check if the instance is already running in the group
    existing_masters = get_existing_instance(conn, opts, die_on_error=False)
    if existing_masters:
        print >> sys.stderr, ("ERROR: There are already instances running \
                               in group %s" % master_group.name)
        sys.exit(1)

    print "Launching instances..."

    try:
        image = conn.get_all_images(image_ids=[opts.ami])[0]
    except:
        print >> sys.stderr, "Could not find AMI " + opts.ami
        sys.exit(1)

    # Launch instance
    master_type = opts.instance_type
    if opts.zone == 'all':
        opts.zone = random.choice(conn.get_all_zones()).name
    master_res = image.run(key_name=opts.key_pair,
                           security_group_ids=[master_group.id],
                           instance_type=master_type,
                           placement=opts.zone,
                           min_count=1,
                           max_count=1,
                           placement_group=None,
                           user_data=user_data_content)

    master_nodes = master_res.instances
    print "Launched instance in %s, regid = %s" % (opts.zone, master_res.id)

    print "Waiting for AWS to propagate instance metadata..."
    time.sleep(5)
    # Give the instance a descriptive name
    for master in master_nodes:
        master.add_tag(
            key='Name',
            value='{cn}-{iid}'.format(cn=opts.instance_name, iid=master.id))

    # Return all the instances
    return master_nodes


# Get the EC2 security group of the given name, creating it if it doesn't
# exist
def get_or_make_group(conn, name, vpc_id):
    groups = conn.get_all_security_groups()
    group = [g for g in groups if g.name == name]
    if len(group) > 0:
        return group[0]
    else:
        print "Creating security group " + name
        return conn.create_security_group(name, "EC2 instance group", vpc_id)


def wait_for_instance_state(conn, opts, theinstance, instance_state):
    """
    Wait for the instance to reach a designated state.

    theinstance: the launched instance: boto.ec2.instance.Instance
    instance_state: a string representing the desired state of the instance
           value can be 'ssh-ready' or a valid value from
           boto.ec2.instance.InstanceState such as
           'running', 'terminated', etc.
    """
    sys.stdout.write("Waiting for instance to enter '{s}' \
    state.".format(s=instance_state))
    sys.stdout.flush()

    start_time = datetime.now()
    num_attempts = 0

    while True:
        time.sleep(2 * num_attempts)  # seconds

        theinstance.update()
        status = conn.get_all_instance_status(instance_ids=theinstance.id)

        if instance_state == 'ssh-ready':
            if theinstance.state == 'running' and \
               status[0].system_status.status == 'ok' and \
               status[0].instance_status.status == 'ok' and \
               is_ssh_available(theinstance.public_dns_name, opts):
                break
        else:
            if theinstance.state == instance_state:
                break

        num_attempts += 1
        sys.stdout.write(".")
        sys.stdout.flush()

    sys.stdout.write("\n")

    print "Instance is now in '{s}' state. Waited {t} seconds.".format(
        s=instance_state,
        t=(datetime.now() - start_time).seconds
    )


def is_ssh_available(host, opts, print_ssh_output=True):
    """
    Check if SSH is available on a host.
    """
    try:
        ssh_connect(host, opts)
    except (paramiko.ssh_exception.BadHostKeyException,
            paramiko.ssh_exception.AuthenticationException,
            paramiko.ssh_exception.SSHException,
            socket.error) as ssh_exception:
        ssh_success = False
        if print_ssh_output:
            print "Warning: SSH connection error. (This could be temporary.)"
            print "Host: " + host
            print "Error: " + ssh_exception
    else:
        ssh_success = True

    return ssh_success


def allow_root_ssh(ssh_client):
    print_status("Allowing SSH access as root")
    ssh(ssh_client, 'sudo mkdir -p /root/.ssh && '
                    'sudo chmod 700 /root/.ssh && '
                    'sudo cp /home/ubuntu/.ssh/authorized_keys /root/.ssh/ && '
                    'sudo chmod 600 /root/.ssh/authorized_keys && '
                    'sudo service ssh restart')

    print_success()


def setup_analysis(ssh_client):
    print_status("Copying python script to run analysis")
    ssh(ssh_client, 'mkdir -p /mnt/analysis/DATA')
    ssh_sftp = ssh_client.open_sftp()
    ssh_sftp.put('./runanalysis.py', 'runanalysis.py')
    # Defining path for remote destination doesn't work in Windows?
    ssh_sftp.close()
    ssh(ssh_client, 'mv runanalysis.py /mnt/analysis/DATA &&'
                    'chmod 700 /mnt/analysis/DATA/runanalysis.py')


def mount_volume(ssh_client):
    print_status("Mounting ephemeral storage to /mnt/analysis")
    ssh(ssh_client, 'mkfs /dev/xvdb')
    ssh(ssh_client, 'mkdir -p /mnt/analysis')
    ssh(ssh_client, 'mount /dev/xvdb /mnt/analysis')
    print_success()


def setup_ipythonnotebook(ssh_client, master):
    print_status("Copying ipython notebook setup script")
    ssh_sftp = ssh_client.open_sftp()
    ssh_sftp.put('./setup-notebook-ec2.sh', 'setup-notebook-ec2.sh')
    ssh_sftp.close()
    ssh(ssh_client, "chmod 700 setup-notebook-ec2.sh")

    print("    Login to the instance and run ./setup-notebook-ec2.sh to \
               finish iPython Notebook setup")
    print "    Then access it at " + colored("https://%s:8888" % master,
                                             'blue')


def launch_ec2(conn, opts):
    opts.zone = random.choice(conn.get_all_zones()).name

    if opts.resume:
        master_nodes = get_existing_instance(conn, opts)
        print "Instance at " + master_nodes[0].public_dns_name
    else:
        master_nodes = launch_instance(conn, opts)
        print "Instance at " + master_nodes[0].public_dns_name

        wait_for_instance_state(theinstance=(master_nodes[0]),
                                instance_state='ssh-ready',
                                opts=opts, conn=conn)
    print("")

    ssh_client = ssh_connect(master_nodes[0].public_dns_name, opts)
    if not opts.copyscript:
        allow_root_ssh(ssh_client)
        # After restarting ssh daemon in setup_instance, you need to get the
        # ip address again. I (Vijay) don't understand this, but getting it
        # again works.
        master_nodes = get_existing_instance(conn, opts, outputflag=False)
        master = master_nodes[0].public_dns_name
        opts.user = "root"
        ssh_client.close()
        ssh_client.connect(master_nodes[0].public_dns_name,
                           username=opts.user,
                           key_filename=opts.identity_file)
        mount_volume(ssh_client)
        setup_sign_of_life(ssh_client)
        # this is to make sure the instance sends a sign of life signal to
        # your local computer; without this, the sshd closes the connection
        # over the analysis and the local terminal freezes
        install_anaconda(ssh_client)
        install_opencv(ssh_client)
        install_picos(ssh_client)
        install_MDP(ssh_client)
        install_sima(ssh_client)
        setup_analysis(ssh_client)
        if opts.action == 'launch':
            setup_ipythonnotebook(ssh_client, master)
    else:
        master_nodes = get_existing_instance(conn, opts, outputflag=False)
        master = master_nodes[0].public_dns_name
        opts.user = "root"
        ssh_client.close()
        ssh_client.connect(master_nodes[0].public_dns_name,
                           username=opts.user,
                           key_filename=opts.identity_file)
        setup_analysis(ssh_client)

    print "Instance successfully launched!"
    return ssh_client, master


def set_up_screen(ssh_client):
    # sets up a dtach session for the runanalysis script
    ssh(ssh_client, 'sudo apt-get install dtach')
    ssh(ssh_client, "rm -f /mnt/analysis/DATA/runanalysis.sh &&"
                    "echo '#!/usr/bin/env bash\npython \
                     /mnt/analysis/DATA/runanalysis.py' \
                     >> /mnt/analysis/DATA/runanalysis.sh &&"
                    "chmod 700 /mnt/analysis/DATA/runanalysis.sh")


# The function below is called every timeout period to make sure the
# instance hasn't dropped the internet connection while analysis is
# running
def get_back_to_analysis(ssh_client, conn, opts):
    print('Reconnecting to the analysis output stream')
    try:
        ssh(ssh_client, "dtach -a /tmp/runanalysis -r winch",
            get_pty=True, display=True)
    except AnalysisError as e:
        print >> sys.stderr, (e)
        if not opts.terminate_on_error:
            sys.exit(1)
    except TimeoutError:
        print('Reconnecting session so instance does\
              not drop connection')
        get_back_to_analysis(ssh_client, conn, opts)
    except KeyboardInterrupt:
        print('Keyboard interrupt received. The analysis will \
              continue; you can still reconnect')
        sys.exit(1)
    except:
        cmd = 'if [ -f /tmp/analysisfailed.txt ]; then\n\
               echo "Analysis Failed"\n\
               elif [ -f /tmp/analysissuccess.txt ]; then\n\
               echo "Analysis Success!"\n\
               else\n\
               echo "File not found"\n\
               fi'
        temp = ssh(ssh_client, cmd, display=True)
        if 'Failed' in temp:
            print('Analysis failed')
            if not opts.terminate_on_error:
                sys.exit(1)
        elif 'Success' in temp:
            print('Analysis success!')
            sys.exit(1)
        else:
            print('There is no previous session to which you can \
                   reattach!')
            if not opts.terminate_on_error:
                sys.exit(1)
    else:
        print("Analysis successfully completed!")

    if 'terminate' in opts.action:
        print("Terminating the EC2 instance")
        (master_nodes) = get_existing_instance(conn, opts,
                                               die_on_error=False)
        for inst in master_nodes:
            inst.terminate()
        print("Terminated!")
        sys.exit(1)
    else:
        sys.exit(1)


def main():
    parser = ArgumentParser(description="Create connection to Amazon EC2",
                            formatter_class=RawTextHelpFormatter)
    parser.add_argument("action", help="Use one of the following options:\n"
                        "1. launch             : launches the instance, " +
                        "copies\n"
                        "                        analysis and ipython " +
                        "notebook\n"
                        "                        scripts.\n"
                        "                        Manually log in to run " +
                        "ipython\n"
                        "                        notebook or the analysis " +
                        "script.\n"
                        "                        If instance already exists," +
                        " thro\n"
                        "                        ws an error unless the " +
                        "--resume\n"
                        "                        flag is specified.\n"
                        "2. analyze            : automatically runs the " +
                        "runanalys\n"
                        "                        is.py script and shows " +
                        "you the \n"
                        "                        output. Does not terminate " +
                        "after\n"
                        "                        running.\n"
                        "                        If instance already exists," +
                        " thro\n"
                        "                        ws an error unless the " +
                        "--resume\n"
                        "                        flag is specified.\n"
                        "3. analyzeandterminate: automatically runs the " +
                        "runanalys\n"
                        "                        is.py script and shows " +
                        "you the \n"
                        "                        output. Terminates if " +
                        "analysis \n"
                        "                        successfully completed. " +
                        "If analy\n"
                        "                        sis failed, terminates " +
                        "if \n"
                        "                        --terminate_on_error flag " +
                        "is \n"
                        "                        specified.\n"
                        "                        If instance already exists," +
                        " thro\n"
                        "                        ws an error unless the " +
                        "--resume\n"
                        "                        flag is specified.\n"
                        "4. login              : Login to the instance\n"
                        "5. reconnect          : actions <analyze> and \n"
                        "                        <analyzeandterminate> run " +
                        "a \n"
                        "                        detached session, meaning " +
                        "that\n"
                        "                        if your terminal " +
                        "accidentally \n"
                        "                        shuts down or you log out " +
                        "or \n"
                        "                        lose connection, the " +
                        "analysis \n"
                        "                        still continues on the EC2 " +
                        "insta\n"
                        "                        nce. You should use " +
                        "reconnect if\n"
                        "                        you want to reattach to " +
                        "a prev\n"
                        "                        ious <analyze> session.\n"
                        "6. reconnectandterminate:Similar to reconnect but " +
                        "mimics\n"
                        "                        <analyzeandterminate> " +
                        "behavior.\n"
                        "                        So if analysis successfully\n"
                        "                        completes, terminates " +
                        "instance.\n"
                        "                        If there was an error, " +
                        "terminat\n"
                        "                        es only if " +
                        "--terminate_on_error\n"
                        "                        flag is specified. For " +
                        "reconnect\n"
                        "                        and reconectandterminate, " +
                        "it doe\n"
                        "                        s not matter if you ran " +
                        "analyze\n"
                        "                        or analyzeandterminate. " +
                        "So if yo\n"
                        "                        u ran analyze previously " +
                        "and the\n"
                        "                        terminal detached, you can " +
                        "do\n"
                        "                        reconnectandterminate and " +
                        "it'll\n"
                        "                        give you " +
                        "analyzeandterminate\n"
                        "                        behavior for the " +
                        "reattached\n"
                        "                        session."
                        "7. terminate          : terminates the instance\n"
                        "8. start              : starts a previously stopped" +
                        " inst\n"
                        "                        ance\n"
                        "9. stop               : stop an instance. This " +
                        "will \n"
                        "                        remove all downloaded data " +
                        "but \n"
                        "                        the installs are kept. " +
                        "You will\n"
                        "                        have to pay a low rate to " +
                        "Amazon\n"
                        "                        for keeping an instance " +
                        "alive but\n"
                        "                        stopped.")

    parser.add_argument("instance_name", help="Name of EC2 instance")
    parser.add_argument("-k", "--key_pair",
                        default="mykey", help="Key pair to use on instances")
    parser.add_argument("-i", "--identity_file",
                        default=os.path.join(os.getcwd(), "mykey.pem"),
                        help="SSH private key file to use for logging into " +
                              "instances",)
    parser.add_argument("-r", "--region", default="us-east-1",
                        help="EC2 region to launch instances " +
                              "(default: us-east-1)")
    parser.add_argument("-t", "--instance_type", default="r3.2xlarge",
                        help="Type of instance to launch " +
                              "(default: r3.2xlarge).\n"
                             "WARNING: must be 64-bit; "
                             "small instances won't\n work. Pick instances " +
                              "with at least one ephemeral storage device")
    parser.add_argument("--resume", default=False, action="store_true",
                        help="If this flag is present, resumes\n"
                             "installation on a previously launched " +
                             "instance\n"
                             "Use if the instance has already been launched")
    parser.add_argument("--copyscript", default=False, action="store_true",
                        help="If this flag is present, the script assumes " +
                             "that\n"
                             "installations have already been done. So it " +
                             "doesn't\n"
                             "perform any installations. If action==launch,\n"
                             "it copies runanalysis.py. So, use this if you " +
                             "changed\n"
                             "something in the script locally and you " +
                             "want it\n"
                             "copied over again to the instance.\n"
                             "If action==analyze(andterminate), the script " +
                             "copies\n"
                             "runanalysis.py, runs it (and then terminates\n"
                             "the instance)")
    parser.add_argument("--terminate_on_error", default=False,
                        action="store_true",
                        help="Terminates the EC2 instance if runanalysis.py\n"
                             "returns an error. Will not terminate without\n"
                             "this flag if there was an error.")
    opts = parser.parse_args()

    opts.user = "ubuntu"
    # Login first as ubuntu and then copy ssh authorization files for
    # root login
    opts.ami = "ami-d05e75b8"  # Ubuntu AMI from EC2 console
    opts.vpc_id = None
    opts.authorized_address = "0.0.0.0/0"
    opts.ebs_vol_size = 0

    # Check file permissions for the key file. This is not an issue on Windows
    if oct(os.stat(opts.identity_file).st_mode & 0777) != oct(0600):
        raise Exception('Set correct permissions for the key file. ' +
                        'It should be 0600!')

    # Create EC2 connection
    try:
        conn = ec2.connect_to_region(opts.region)
    except Exception as e:
        print >> sys.stderr, (e)
        sys.exit(1)

    if opts.action == "launch":
        ssh_client, master = launch_ec2(conn, opts)
        print("Login to the instance and run python runanalysis.py \
               to perform analysis")
        print("")
        print("Public IP for instance is " + colored("http://%s" % master,
                                                     'blue'))
        print("")
    elif opts.action == "analyzeandterminate" or opts.action == "analyze":
        ssh_client, master = launch_ec2(conn, opts)
        print("")
        print("Running analysis now. Access log file on S3")
        print("")

        print_status('Setting up a detachable session')
        set_up_screen(ssh_client)
        with open("analysislog.txt", "w") as myfile:
            myfile.write("Starting analysis now at time\n")
            t = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
            myfile.write(t)
            myfile.write("\n")
        try:
            ssh(ssh_client, "dtach -c /tmp/runanalysis -r winch\
                /mnt/analysis/DATA/runanalysis.sh", get_pty=True,
                display=True)
        except AnalysisError as ae:
            print >> sys.stderr, (ae)
            if not opts.terminate_on_error:
                sys.exit(1)
        except KeyboardInterrupt:
            print('Keyboard interrupt received. The analysis will continue; \
                   you can still reconnect')
            sys.exit(1)
        except TimeoutError:
            get_back_to_analysis(ssh_client, conn, opts)
        except:
            print('Unknown error')
            if not opts.terminate_on_error:
                sys.exit(1)
        else:
            print("Analysis successfully completed!")

        if opts.action == "analyzeandterminate":
            print("Terminating the EC2 instance")
            (master_nodes) = get_existing_instance(conn, opts,
                                                   die_on_error=False)
            for inst in master_nodes:
                inst.terminate()
            print("Terminated!")
            sys.exit(1)
        else:
            sys.exit(1)
    else:
        (master_nodes) = get_existing_instance(conn, opts)
        master = master_nodes[0].public_dns_name
        opts.user = "root"

        # Login to the instance
        if opts.action == "login":
            if os.name == 'nt':
                print "Logging into instance " + master + " as " +\
                       opts.user + " ..."
                subprocess.check_call(['plink', '-i',
                                       opts.identity_file.replace(".pem",
                                                                  ".ppk"),
                                       '-ssh', "%s@%s" % (opts.user, master)])
                # use key with .ppk extension for plink connection
            else:
                print "Logging into instance " + master + " as " \
                       + opts.user + " ..."
                subprocess.check_call(['ssh', '-i', opts.identity_file,
                                       '-o', 'StrictHostKeyChecking=no',
                                       '-o', 'UserKnownHostsFile=/dev/null',
                                       '-o', 'CheckHostIP=no',
                                       '-o', 'LogLevel=quiet',
                                       '-t', '-t', "%s@%s" % (opts.user,
                                                              master)])

        # Stop instance
        elif opts.action == "stop":
            response = raw_input(
                "Are you sure you want to stop the instance " +
                opts.instance_name + "?\n"
                "DATA ON EPHEMERAL DISKS WILL BE LOST, BUT THE INSTANCE \
                 WILL KEEP USING SPACE ON\n"
                "AMAZON EBS IF IT IS EBS-BACKED!!\n" +
                "Stop instance " + opts.instance_name + " (y/N): ")
            if response == "y":
                (master_nodes) = get_existing_instance(conn, opts,
                                                       die_on_error=False)
                print "Stopping instance..."
                for inst in master_nodes:
                    if inst.state not in ["shutting-down", "terminated"]:
                        inst.stop()

        # Restart a stopped instance
        elif opts.action == "start":
            print "Starting instance..."
            for inst in master_nodes:
                if inst.state not in ["shutting-down", "terminated"]:
                    inst.start()
            wait_for_instance_state(conn=conn, instance_state='ssh-ready',
                                    opts=opts, theinstance=master_nodes[0])
            print("")
            print("")
            print("Instance successfully restarted!")
            print("")
            (master_nodes) = get_existing_instance(conn, opts)
            master = master_nodes[0].public_dns_name
            opts.user = "root"
            ssh_client = ssh_connect(master, opts)
            mount_volume(ssh_client)

        # Terminate the instance
        elif opts.action == "terminate":
            response = raw_input("Are you sure you want to terminate the \
                                  instance " + opts.instance_name + "?\n"
                                 "ALL DATA WILL BE LOST!!\n"
                                 "Terminate instance " + opts.instance_name +
                                 " (y/N): ")
            if response == "y":
                (master_nodes) = get_existing_instance(conn, opts,
                                                       die_on_error=False)
                print "Terminating instance..."
                for inst in master_nodes:
                    inst.terminate()

        elif (opts.action == "reconnect" or
              opts.action == "reconnectandterminate"):
            ssh_client = ssh_connect(master, opts)
            get_back_to_analysis(ssh_client, conn, opts)

        else:
            raise NotImplementedError(
                  "action: " + opts.action + " not recognized")


if __name__ == "__main__":
    main()
